import random
import json
import pandas as pd
import torch
import torch.nn as nn 

import matplotlib.pyplot as plt
import os
import numpy as np
import time
import pickle
np.set_printoptions(precision=4, suppress=True) 


# import custom files
from S2S import *
from sim import *
from data_utils import *
from model_structure_param import *


loss_fn = nn.MSELoss(reduction = 'none')

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

def plot(price_hist, account_value_hist, buy_decisions, sell_decisions, stock_growth, account_growth, ax1, ax2, annotations, show = False):
    
    ax1.clear()
    ax2.clear()

    price_hist_start = price_hist[0]
    account_value_start = account_value_hist[0]


    # calculate percentage change for both lines
    y1_pct_change = [(i-price_hist_start)/price_hist_start for i in price_hist]
    y1_pct_change_min = min(y1_pct_change)
    y1_pct_change_max = max(y1_pct_change)
    y2_pct_change = [(i-account_value_start)/account_value_start for i in account_value_hist]
    y2_pct_change_min = min(y2_pct_change)
    y2_pct_change_max = max(y2_pct_change)

    pct_change_min = min(y1_pct_change_min, y2_pct_change_min)
    pct_change_max = max(y1_pct_change_max, y2_pct_change_max)

    ax1_y_min = price_hist_start*(1+pct_change_min - 0.05)
    ax1_y_max = price_hist_start*(1+pct_change_max + 0.05)

    ax2_y_min = account_value_start*(1+pct_change_min - 0.05)
    ax2_y_max = account_value_start*(1+pct_change_max + 0.05)

    # set limits of both y-axes to begin at the same height
    
    ax1.set_ylim([ax1_y_min, ax1_y_max])
    ax2.set_ylim([ax2_y_min, ax2_y_max])

    
    buy_time = [n for n, b in enumerate(buy_decisions) if b != 0]
    buy_price = [p for x, p in zip(buy_decisions,price_hist) if x != 0]

    sell_time = [n for n, s in enumerate(sell_decisions) if s != 0]
    sell_price = [p for x, p in zip(sell_decisions,price_hist) if x != 0]

    

    ax1.plot(price_hist, label = 'price', color = 'b')
    ax1.plot(sell_time, sell_price, marker = 'v', label = 'sell', color = 'y')
    ax1.plot(buy_time, buy_price, marker = '^', label = 'buy', color = 'g')
    ax2.plot(account_value_hist, label='account value', color = 'r')


    ax1_annotation = (f"{stock_growth:.2f}%", (len(account_value_hist), price_hist[-1]))
    ax2_annotation = (f"{account_growth:.2f}%", (len(account_value_hist), account_value_hist[-1]))
    annotations.append((ax1_annotation, ax2_annotation))
    for ax1_annotation, ax2_annotation in annotations:
        ax1.annotate(ax1_annotation[0], xy=ax1_annotation[1], xytext=(10, -20), textcoords="offset points",
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2"))
        ax2.annotate(ax2_annotation[0], xy=ax2_annotation[1], xytext=(10, -20), textcoords="offset points",
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2"))

    ax1.legend(loc='upper right')
    ax2.legend(loc='upper left')
    plt.pause(0.1)
    if show:
        plt.show()

def back_test(model, data_loader, col_names, num_epochs = 1, block_col = None, to_plot = True, to_print = True):
    timers = [0.0] * 10

    st = time.time()
    teacher_forcing_ratio = 0
    model.eval()
    start_time = time.time()

    average_loss = 0

    arr = np.ones(prediction_window)
    for i in range(1, prediction_window):
        arr[i] = arr[i-1] * weight_decay
    weights = arr.reshape(prediction_window,1)

    decisions = []
    buy_decisions = []
    sell_decisions = []
    account_value_hist = []
    price_hist = []
    start_price = 0
    start_balance = 0
    end_price = 0
    
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    timers[0] = time.time() - st # time spent initializing
    st = time.time()
    for epoch in range(num_epochs):
        i=0
        annotations = []
        for i, (x_batch, y_batch, x_raw_close) in enumerate(data_loader):

            timers[1] += time.time() - st # time spent using data_loader
            st = time.time()

            # print('x_batch.shape: ', x_batch.shape)
            # print('y_batch.shape: ', y_batch.shape)
            # x_batch   [N, hist_window, feature_num]
            # y_batch & y_pred  [N, prediction_window]
            # note here N = 1
            if block_col != None:
                x_batch[:,:,block_col] = 0

            x_batch = x_batch.float().to(device) 
            # probably need to do numpy to pt tensor in the dataset; need testing on efficiency 
            # !!! CAN"T BE DONE. before dataloarder they are numpy array, not torch tensor
            # print("one input: ", x_batch[0:,:,:])
            y_batch = y_batch.float().to(device)
        
            y_pred = model(x_batch, None, teacher_forcing_ratio) # [N, prediction_window]
            timers[2] += time.time() - st # time spent on prediction 
            st = time.time()

            loss = loss_fn(y_pred, y_batch)
            loss_val = loss.mean().item()
            average_loss += loss_val
            
            price = x_raw_close.item()
            if (epoch == 0 and i == 0):
                start_price = price
                start_balance = account.evaluate()
            if (epoch == num_epochs-1 and i == len(data_loader)-1):
                end_price = price

            # decision = policy.decide('AAPL', x_batch.clone().detach().cpu(), price, None, account, col_names) # naivemacd
            prediction = y_pred.clone().detach().cpu()
            weighted_prediction = (prediction * weights).sum() / weights.sum()
            decision = policy.decide('AAPL', x_batch.clone().detach().cpu(), price, weighted_prediction, col_names)
            #policy.decide('BABA', None, price, y_pred, account)
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

            account_value = account.evaluate()
            
            account_value_hist.append(account_value)
            price_hist.append(price)

            timers[3] += time.time() - st # time spent on decision making
            st = time.time()

            if (decision[0] != 'n' and to_print) or i == (len(data_loader)-1):
                long_count, profitable_long_count, \
                short_count, profitable_short_count, \
                mean_long_profit_pct, mean_short_profit_pct = policy.get_trade_stat()

                prediction_stat_str = f'decision: {decision[0]}:{decision[1]:>4}, ' + \
                        f'price: {price:>6.2f}, ' + \
                        f'long: {long_count:>4}, ' + \
                        f'\u2713 long: {profitable_long_count:>4}, ' + \
                        f'\u2713 long pct: {profitable_long_count/long_count*100:>5.2f}%, ' + \
                        f'long profit pct: {mean_long_profit_pct:>5.3f}%, ' # + \
                        # f'short: {short_count:>3}, ' + \
                        # f'\u2713 short: {profitable_short_count:>3}, ' + \
                        # f'\u2713 short pct: {profitable_short_count/short_count*100:>5.2f}%, ' + \
                        # f'short profit pct: {mean_short_profit_pct:>5.3f}%'
                

                account_growth = account_value/start_balance*100-100
                stock_growth = price/start_price*100-100

                account_n_stock_str = f'Account Value: {account_value:>10.2f}, ' + \
                      f'accont growth: {account_growth:>6.2f}%, ' + \
                      f'stock growth: {stock_growth:>6.2f}%, ' +  \
                      f'growth diff: {account_growth-stock_growth:>6.2f}%' # + \
                    #   f'past 1000 interval growth: '

                if to_print == True:
                    print(prediction_stat_str)
                    print(account_n_stock_str)
            
            timers[4] += time.time() - st # time spent on printing
            st = time.time()

            draw_interval = 2000
            if (i%draw_interval == 0) and (i != 0) and to_plot == True:
                # start_time_plot = time.time()
                plot(price_hist, account_value_hist, buy_decisions, sell_decisions, stock_growth, account_growth, ax1, ax2, annotations)

                # print(f'plotting completed in {time.time()-start_time_plot:.2f} seconds')
            # print(account.evaluate())

            timers[5] += time.time() - st # time spent on plotting
            st = time.time()
        
        plot(price_hist, account_value_hist, buy_decisions, sell_decisions, stock_growth, account_growth, ax1, ax2, annotations, show = to_plot)
        average_loss /= i
        
        # print(f'Time spent on initialization: {timers[0]:.2f} seconds\n' + \
        #         f'Time spent on data loading: {timers[1]:.2f} seconds\n' + \
        #         f'Time spent on prediction: {timers[2]:.2f} seconds\n' + \
        #         f'Time spent on decision making: {timers[3]:.2f} seconds\n' + \
        #         f'Time spent on printing: {timers[4]:.2f} seconds\n' + \
        #         f'Time spent on plotting: {timers[5]:.2f} seconds\n'
        #       )
        # print(f'Epoch {epoch+1:3}/{num_epochs:3}, ' +
        #       f'Time per epoch: {(time.time()-start_time)/(epoch+1):.2f} seconds, ')
        
        


    print(f'back test completed in {time.time()-start_time:.2f} seconds')

    return  buy_decisions, \
            sell_decisions, \
            account_value_hist, \
            price_hist, \
            start_price, \
            end_price, \
            (prediction_stat_str, account_n_stock_str), \
            average_loss
            # raw_predictions.cpu().numpy(), \
            # raw_targets.cpu().numpy(), \

def locate_cols(strings_list, substring):
    return [i for i, string in enumerate(strings_list) if substring in string]

def save_result(pkl_path, block_str_lst = [], end_strs_lst = [], loss_lst = []):
    print('saving results...')
    with open(pkl_path, 'wb') as f:
        pickle.dump((block_str_lst, end_strs_lst, loss_lst), f)
    
    print('results saved')

if __name__ == "__main__":

    close_idx = 3 # after removing time column


    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Make predictions
    start_time = time.time()
    print("Making Prediction")
    # data_path = '../data/csv/bar_set_huge_20200101_20230417_AAPL_macd_n_time_only.csv'
    # data_path = '../data/csv/bar_set_huge_20230418_20230501_AAPL_23feature.csv'
    # data_path = '../data/csv/bar_set_huge_20200101_20230417_AAPL_indicator.csv'
    time_str = '20230101_20230501'
    name = 'MSFT'
    data_type = '23feature'
    data_path = f'../data/csv/bar_set_huge_{time_str}_{name}_{data_type}.csv'
    pkl_path = 'lists_no_multithread_AAPL_noblock.pkl'

    test_loader, col_names = \
        load_n_split_data(data_path, 
                          hist_window, 
                          prediction_window, 
                          batch_size, 
                          train_ratio = 0, 
                          normalize = True,
                          test = True)
    model = Seq2Seq(input_size, hidden_size, num_layers, output_size, prediction_window, dropout, device).to(device)
    model_pth = f'../model/model_{config_name}.pt'
    model.load_state_dict(torch.load(model_pth))

    block_str_lst = []
    end_strs_lst = []
    loss_lst = []

    try:
        with torch.no_grad():
            # for x in range(feature_num):
            x = [0,1]
            
            account = Account(initial_capital, ['AAPL'])
            policy = SimpleLongShort(account)
            block_str = f'blocking column {x}:{col_names[x]}'
            print(block_str)
            buy_decisions, sell_decisions, account_value_hist, price_hist, start_price, end_price, end_strs, loss= \
                back_test(model, test_loader, col_names, num_epochs = 1, block_col = x, to_plot = True, to_print = True)
            
            block_str_lst.append(block_str)
            end_strs_lst.append(end_strs)
            loss_lst.append(loss)
            # print(f'account value: {account_value_hist[-1]:.2f}')
            # print(f'account growth: {account_value_hist[-1]/initial_capital*100 - 100:.2f}%')
            # print(f'stock value change: {end_price/start_price*100 - 100:.2f}%')

            print(f'Test completed in {time.time()-start_time:.2f} seconds')

        # save_result(pkl_path, block_str_lst, end_strs_lst, loss_lst)
        # plot(predictions, targets, test_size)
        # plot(raw_predictions, raw_targets, test_size)
    except KeyboardInterrupt or Exception or TypeError:
        # save_result(pkl_path, block_str_lst, end_strs_lst, loss_lst)
        pass