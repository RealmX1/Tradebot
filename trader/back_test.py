import random
import json
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch # PyTorch
import torch.nn as nn # PyTorch neural netback_test module
from torch.utils.data import Dataset, DataLoader # PyTorch data utilities
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import AdamW, SGD
# from apex.optimizers import FusedLAMB

import matplotlib.pyplot as plt
import os
import numpy as np
import time
np.set_printoptions(precision=4, suppress=True) 


# import custom files
from S2S import *
from sim import *
from data_utils import *

def get_direction_diff(y_batch,y_pred):
    # true_direction = y_batch-x_batch[:,-1,close_idx:close_idx+1]
    true_direction = np.clip(y_batch.cpu(),0,np.inf) # this turns negative to 0, positive to 1
    true_direction[true_direction != 0] = 1
    # pred_direction = y_pred-x_batch[:,-1,close_idx:close_idx+1]
    pred_direction = np.clip(y_pred.clone().detach().cpu(),0,np.inf)
    pred_direction[pred_direction != 0] = 1
    # print("True: ", true_direction.shape)
    # print("Pred: ", pred_direction)

    instance_num =  true_direction.shape[0]
    prediction_min = true_direction.shape[1]

    total_cells = instance_num * prediction_min
    same_cells = np.count_nonzero(true_direction == pred_direction)

    total_cells_list = np.full((prediction_min,), instance_num)
    same_cells_list = np.count_nonzero(true_direction == pred_direction, axis = 0)
    # print("total_cells: ",total_cells)
    # print("same_cells.shape: ",same_cells.shape)

    return total_cells, same_cells, total_cells_list, same_cells_list


def back_test(model, data_loader, num_epochs = 1):    
    teacher_forcing_ratio = 0
    model.eval()
    start_time = time.time()
    same_cells = 0
    
    predictions = None
    raw_predictions = None
    targets = None
    raw_targets = None
    raw_closes = None

    total_predictions = np.zeros(prediction_window) # one elemnt for each minute of prediction window
    total_true_predictions = np.zeros(prediction_window)

    average_loss = 0

    inverse_mask = torch.linspace(1, 11, 10)
    weights = torch.linspace(1, 0.1, steps=prediction_window).to(device)

    ma_loss = 0
    decisions = []
    buy_decisions = []
    sell_decisions = []
    account_value_hist = []
    price_hist = []
    for epoch in range(num_epochs):
        epoch_predictions = 0
        epoch_true_predictions = 0
        i=0
        for i, (x_batch, y_batch, x_raw_close) in enumerate(data_loader):
            ma_loss *= 0.8
            # print("x_batch[0,:,:]: ", x_batch[0,:,:])
            # x_batch   [N, hist_window, feature_num]
            # y_batch & output  [N, prediction_window]
            x_batch = x_batch.float().to(device) # probably need to do numpy to pt tensor in the dataset; need testing on efficiency #!!! CAN"T BE DONE. before dataloarder they are numpy array, not torch tensor
            # print("one input: ", x_batch[0:,:,:])
            y_batch = y_batch.float().to(device)
            
            y_pred = model(x_batch, y_batch, teacher_forcing_ratio) # [N, prediction_window]
            # print("y_pred.shape: ", y_pred.shape)
            # print("y_batch.shape: ", y_pred.shape)
            # print("y_batch: ", y_batch[0,:])
            
            total_cells, same_cells, total_cells_list,same_cells_list = get_direction_diff(y_batch, y_pred)
            epoch_predictions += total_cells
            epoch_true_predictions += same_cells
            total_predictions += total_cells_list
            total_true_predictions += same_cells_list

            price = x_raw_close.item()
            decision = policy.decide('BABA', None, price, y_pred, account)
            decisions.append(decision)
            if decision[0] == 'b':
                buy_decisions.append(1)
                sell_decisions.append(0)
            elif decision[0] == 's':
                buy_decisions.append(0)
                sell_decisions.append(1)
            else:
                buy_decisions.append(0)
                sell_decisions.append(0)
            account_value_hist.append(account.evaluate())
            price_hist.append(price)
            # print(account.evaluate())
            # mean = mean.float().to(device)
            # std = std.float().to(device)
                
            # raw_prediction = y_pred[:,plot_minutes]
            # prediction = raw_prediction * std[:,close_idx:close_idx+1] + mean[:,close_idx:close_idx+1]
            # raw_target = y_batch[:,plot_minutes]
            # target = raw_target * std[:,close_idx:close_idx+1] + mean[:,close_idx:close_idx+1]

            
            if raw_predictions is None:
                # predictions = prediction
                # raw_predictions = raw_prediction
                # targets = target
                # raw_targets = raw_target
                raw_closes = x_raw_close
            else:
                # predictions = torch.cat((predictions, prediction), dim=0)
                # raw_predictions = torch.cat((raw_predictions, raw_prediction), dim=0)
                # targets = torch.cat((targets, target), dim=0)
                # raw_targets = torch.cat((raw_targets, raw_target), dim=0)
                raw_closes = torch.cat((raw_closes, x_raw_close), dim=0)
            
        correct_direction = epoch_true_predictions / epoch_predictions * 100
            
        print(f'Epoch {epoch+1:3}/{num_epochs:3}, ' +
              f'Time per epoch: {(time.time()-start_time)/(epoch+1):.2f} seconds, ' +
              f'Correct Direction: {correct_direction:.2f}%, ')
            #   + f'Encocder LR: {get_current_lr(optimizers[0]):9.10f}, Decoder LR: {get_current_lr(optimizers[1]):9.10f}') 
        
    print(f'completed in {time.time()-start_time:.2f} seconds')
    average_loss /= num_epochs
    accuracy_list = total_true_predictions / total_predictions * 100
    accuracy_list_print = [round(x, 2) for x in accuracy_list]
    print("Accuracy List: ", accuracy_list_print)

    return  buy_decisions, \
            sell_decisions, \
            account_value_hist, \
            price_hist
            # raw_predictions.cpu().numpy(), \
            # raw_targets.cpu().numpy(), \

if __name__ == "__main__":
    policy = NaiveLong()
    initial_capital = 100000
    account = Account(initial_capital, ['BABA'])


    close_idx = 3 # after removing time column


    # Define hyperparameters
    feature_num         = input_size = 23 # Number of features (i.e. columns) in the CSV file -- the time feature is removed.
    hidden_size         = 200    # Number of neurons in the hidden layer of the LSTM
    num_layers          = 4     # Number of layers in the LSTM
    output_size         = 1     # Number of output values (closing price 1~10min from now)
    prediction_window   = 5
    hist_window         = 100 # using how much data from the past to make prediction?
    data_prep_window    = hist_window + prediction_window # +ouput_size becuase we need to keep 10 for calculating loss


    learning_rate   = 0.0001
    batch_size      = 1
    train_percent   = 0
    num_epochs      = 0
    dropout         = 0.1
    teacher_forcing_ratio = 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('account value: ', account.evaluate())

    # Make predictions
    start_time = time.time()
    print("Making Prediction")
    data_path = 'data/aapl_test.csv'
    test_loader = load_n_split_data(data_path, hist_window, prediction_window, batch_size, train_ratio = 0, global_normalization_list = None)
    model_pth = 'lstm_updown_S2S_attention.pt'
    model = Seq2Seq(input_size, hidden_size, num_layers, output_size, prediction_window, dropout, device).to(device)
    model.load_state_dict(torch.load(model_pth))
    with torch.no_grad():
        buy_decisions, sell_decisions, account_value_hist, price_hist = \
            back_test(model, test_loader, num_epochs = 1)
        print(f'account value: {account_value_hist[-1]:.2f}')
        print(f'account growth: {account_value_hist[-1]/initial_capital*100:.2f}%')
        print(f'prediction completed in {time.time()-start_time:.2f} seconds')

        assert len(buy_decisions) == len(sell_decisions) == len(account_value_hist) == len(price_hist)

        start_time = time.time()
        print('plotting...')
        buy_time = [i for i, x in enumerate(buy_decisions) if x != 0]
        buy_price = [p for x, p in zip(buy_decisions,price_hist) if x != 0]

        sell_time = [i for i, x in enumerate(sell_decisions) if x != 0]
        sell_price = [p for x, p in zip(sell_decisions,price_hist) if x != 0]
        fig, ax1 = plt.subplots()

        
        ax1.plot(price_hist, label = 'price')
        ax1.scatter(buy_time, buy_price, marker = '^', label = 'buy', )
        ax1.scatter(sell_time, sell_price, marker = 'v', label = 'sell')

        ax2 = ax1.twinx()
        plt.plot(account_value_hist, label='account value')
        plt.legend()
        print(f'plotting completed in {time.time()-start_time:.2f} seconds')
        plt.show()
        # plot(predictions, targets, test_size)
        # plot(raw_predictions, raw_targets, test_size)
    