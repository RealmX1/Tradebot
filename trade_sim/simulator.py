import random
import numpy as np
import pandas as pd
import torch
from data_prep import *
from agents import Agent

simulation_range = test_size # getting test_size from data_utils.py
model_path = cwd+"/lstm.pt"


def preprocess(price_hist):
    # Perform any necessary preprocessing on price_hist here
    return input_tensor

def simulate_step(agent, input_tensor):
    # Get the agent's action
    action = agent.get_action(input_tensor)

    # Get the stock price
    stock_price = get_stock_price(identifier="")

    

    # Update the agent's account value history
    agent.update_account_value_hist()

    return agent.get_account_value()

def simulate(model_path, window): # should window be a parameter?
    # Create an agent with an initial account balance of $1000 and no stock holdings
    agent = Agent(account_balance = 10000, model_path = model_path)

    for i in range (simulation_range):
        input_tensor = get_stock_price(i,)

        # Simulate a step
        account_value = simulate_step(agent, input_tensor)

if __name__ == '__main__':
    # Load the data and preprocess it
    
    # Run the simulation
    account_value_hist = simulate(model_path=model_path, window=100)



    print(account_value_hist)