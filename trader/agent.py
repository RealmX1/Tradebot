from data_utils import * # need to import "Get_stock_price(identifier="")"
from simulator import *

import random
import numpy as np
import pandas as pd
import torch


class Agent:
    def __init__(self, account_balance, model_path, stock_holding = None, stock_price = 0):
        self.account_balance = account_balance
        self.account_value = account_balance + stock_holding * stock_price
        self.account_value_hist = [account_balance]
        self.stock_holding = stock_holding
        
        self.model = LSTM(input_size, hidden_size, num_layers, output_size, drop_out).to(device)
        if os.path.exists(model_path):
            print("Loading existing model")
            self.model.load_state_dict(torch.load(model_path))
        else:
            print("No existing model")
        self.model = torch.load(model_path)  # Load PyTorch model from provided path
    
    def get_account_value(self):
        return self.account_value
    
    def get_account_value_hist(self):
        return self.account_value_hist

    def buy_stock(self, num_shares=None, percent_account_value=None):
        if num_shares is not None:
            cost = get_stock_price() * num_shares # TODO: get_stock_price() is not defined
        elif percent_account_value is not None:
            cost = self.account_balance * percent_account_value
            num_shares = cost / get_stock_price()

        if cost <= self.account_balance:
            self.account_balance -= cost
            self.stock_holding += num_shares
            print(f"Bought {num_shares} shares of stock at {get_stock_price()} per share for a total of {cost}")
        else:
            print(f"Not enough funds to buy {num_shares} shares of stock at {get_stock_price()} per share")        

    def sell_stock(self, stock_price, num_shares=None, percent_stock_held=None):
        if num_shares is not None:
            revenue = stock_price * num_shares
        elif percent_stock_held is not None:
            revenue = self.stock_holding * percent_stock_held * get_stock_price()
            num_shares = self.stock_holding * percent_stock_held

        if num_shares <= self.stock_holding:
            self.account_balance += revenue
            self.stock_holding -= num_shares
            print(f"Sold {num_shares} shares of stock at {get_stock_price()} per share for a total of {revenue}")
        else:
            print(f"Not enough shares to sell {num_shares} shares of stock at {get_stock_price()} per share")
        

    # this is supposed to be called every simulation step
    def manage_stock(self, input, window):
        # Get past stock prices
        price_hist = get_price_hist(identifier, window)

        # Preprocess price_hist and feed into model to get prediction
        input_tensor = preprocess(price_hist)
        prediction = self.model(input_tensor)

        # Make buy/sell decision based on prediction
        decision = torch.argmax(prediction).item()
        if decision == 0:
            self.buy_stock()
        elif decision == 1:
            self.sell_stock()
        
        # update account value and account value history
        self.acount_value = self.cash + stock_price * self.holdings
        self.account_value_hist.append(self.acount_value)

    





def test_agent():
    agent = Agent(1000, 0, "model.pt")
    for i in range(100):
        agent.manage_stock("NVDA", 100)
        agent.update_account_value_hist()
    print(agent.get_account_value_hist())