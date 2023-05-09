import random
import json
import psutil

import os
import csv
import time
import pickle

import numpy as np
import torch
import torch.nn as nn 
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# np.set_printoptions(precision=4, suppress=True) 

import sys
sys.path.append('AI')  # add the path to my_project
sys.path.append('alpaca_api') 
sys.path.append('sim')


# import custom files
from S2S import *
from policy import *
from account import *
from data_utils import *
from model_structure_param import *
from common import *


loss_fn = nn.MSELoss(reduction = 'none')

complete_log_pth = 'log/complete_test_log/tmp.txt'
result_log_pth = 'log/back_test_result_log.txt'

result_csv_pth = "log/back_test_log.csv"
csv_header = ['symbol', 'blocked_col_name', 'account_value', 'account_growth', 'stock_growth', 'pct_growth_diff', 'interval_per_trade', 'long_count', 'profitable_long_count', 'mean_long_profit_pct', 'occupancy_rate','test_time', 'model_pth', 'data_pth']

data_pth = '/data/csv/wat?'
model_pth = 'model/wat?'



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def get_direction_diff(y_batch,y_pred):
    # true_direction = y_batch-x_batch[:,-1,close_idx:close_idx+1]
    true_direction = np.clip(y_batch.cpu(),0,np.inf) # this turns negative to 0, positive to 1
    true_direction[true_direction != 0] = 1
    # pred_direction = y_pred-x_batch[:,-1,close_idx:close_idx+1]
    pred_direction = np.clip(y_pred.clone().detach().cpu(),0,np.inf)
    pred_direction[pred_direction != 0] = 1
    # print_n_log(complete_log_pth, "True: ", true_direction.shape)
    # print_n_log(complete_log_pth, "Pred: ", pred_direction)

    instance_num =  true_direction.shape[0]
    prediction_min = true_direction.shape[1]

    total_cells = instance_num * prediction_min
    same_cells = np.count_nonzero(true_direction == pred_direction)

    total_cells_list = np.full((prediction_min,), instance_num)
    same_cells_list = np.count_nonzero(true_direction == pred_direction, axis = 0)
    # print_n_log(complete_log_pth, "total_cells: ",total_cells)
    # print_n_log(complete_log_pth, "same_cells.shape: ",same_cells.shape)

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
    plt.pause(0.5)
    if show:
        plt.show()

def time_analysis(timers, pth = None):
    global complete_log_pth
    log_pth = complete_log_pth
    # if pth:
    #     log_pth = pth
    print_n_log(log_pth, f'Time spent on initialization: {timers[0]:.2f} seconds\n' + \
        f'Time spent on data loading: {timers[1]:.2f} seconds\n' + \
        f'Time spent on data prep: {timers[6]:.2f} seconds\n' + \
        f'Time spent on prediction: {timers[2]:.2f} seconds\n' + \
        f'Time spent on decision making: {timers[3]:.2f} seconds\n' + \
        f'Time spent on printing: {timers[4]:.2f} seconds\n' + \
        f'Time spent on plotting: {timers[5]:.2f} seconds\n'
        )  
        

def back_test(policy, model, data_loader, col_names, weights, trade_df = None, trade_data = False, num_epochs = 1, blocked_col = None, blocked_col_name = 'None', to_plot = True, to_print = True):
    global complete_log_pth, result_log_pth
    account = policy.account

    timers = [0.0] * 10

    st = time.time()
    teacher_forcing_ratio = 0
    model.eval()
    start_time = time.time()

    average_loss = 0

    

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
    prev_long_count = 0
    prev_short_count = 0
    prev_interval = 0

    zero_balance_timer = 0 # i.e., has a position on this stock

    step_per_sec = 10
    
    draw_interval = 5000  
    try:
        annotations = []
        for i, (x_batch, y_batch, x_raw_close, timestamp) in enumerate(data_loader):
            
            step_start = time.time()

            timers[1] += time.time() - st # time spent using data_loader
            st = time.time()

            # print_n_log(complete_log_pth, 'x_batch.shape: ', x_batch.shape)
            # print_n_log(complete_log_pth, 'y_batch.shape: ', y_batch.shape)
            # x_batch   [N, hist_window, feature_num]
            # y_batch & y_pred  [N, prediction_window]
            # note here N = 1
            if blocked_col != None:
                x_batch[:,:,blocked_col] = 0

            x_batch = x_batch.float().to(device) 
            # probably need to do numpy to pt tensor in the dataset; need testing on efficiency 
            # !!! CAN"T BE DONE. before dataloarder they are numpy array, not torch tensor
            # print_n_log(complete_log_pth, "one input: ", x_batch[0:,:,:])
            y_batch = y_batch.float().to(device)
            
            timers[6] += time.time() - st # time spent on data prep
            st = time.time()
            y_pred = model(x_batch, None, teacher_forcing_ratio) # [N, prediction_window]
            timers[2] += time.time() - st # time spent on prediction 
            st = time.time()

            loss = loss_fn(y_pred, y_batch)
            loss_val = loss.mean().item()
            average_loss += loss_val
            
            price = x_raw_close.item()
            if i == 0:
                start_price = price
                start_balance = account.evaluate()
            if i == len(data_loader)-1:
                end_price = price

            # decision = policy.decide('AAPL', x_batch.clone().detach().cpu(), price, None, account, col_names) # naivemacd
            prediction = y_pred.clone().detach().cpu().numpy()
            # print_n_log(complete_log_pth, prediction.shape, weights.shape)
            weighted_prediction = (prediction * weights).sum() / weights.sum()
            decision = policy.decide('MSFT', x_batch.clone().detach().cpu(), price, weighted_prediction, col_names)
            
            
            #policy.decide('BABA', None, price, y_pred, account)
            if policy.has_position():
                zero_balance_timer += 1
            decisions.append(decision)
            if decision[0] == 'b':
                if trade_data:
                    trade_rows = trade_df[trade_df.index == timestamp[0]]
                    completed = False
                    for index, row in trade_rows.iterrows():
                        # print_n_log(complete_log_pth, row['price'], price)
                        if row['price'] <= price:
                            print_n_log(complete_log_pth, 'buy order filled')
                            policy.complete_buy_order('MSFT', row['price'])
                            completed = True
                            break
                    
                    if not completed:
                        print_n_log(complete_log_pth, 'buy order not filled; cancelling order...')
                        policy.cancel_buy_order('MSFT', unfilled=True)
                
                buy_decisions.append(1)
                sell_decisions.append(0)
            elif decision[0] == 's':
                if trade_data:
                    trade_rows = trade_df[trade_df.index == timestamp[0]]
                    completed = False
                    for index, row in trade_rows.iterrows():
                        if row['price'] >= price:
                            print_n_log(complete_log_pth, 'sell order filled')
                            policy.complete_sell_order('MSFT', row['price'])
                            completed = True
                            break
                    
                    if not completed:
                        print_n_log(complete_log_pth, 'sell order not filled; cancelling order...')
                        policy.cancel_sell_order('MSFT', unfilled=True)

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
            
            long_count, profitable_long_count, unfilled_buy,\
                short_count, profitable_short_count, unfilled_sell,\
                mean_long_profit_pct, mean_short_profit_pct = policy.get_trade_stat()
            
            account_growth = account_value/start_balance*100-100
            stock_growth = price/start_price*100-100
            pct_growth_diff = (account_growth+100)/(stock_growth+100)*100-100
            if (decision[0] != 'n' and to_print) or i == (len(data_loader)-1):
                # first_print_str = f'decision: {decision[0]}:{decision[1]:>5}, timestamp: {timestamp[0]}'
                # print_n_log(complete_log_pth, first_print_str)

                

                profitable_long_pct = profitable_long_count/long_count*100
                
                occupancy_rate = zero_balance_timer/(i+1)*100

                decision_str = f'decision: {decision[0]}:{decision[1]:>5}, timestamp: {timestamp[0]}, price: {price:>6.2f}, '

                prediction_stat_str = \
                        f'unfilled buy & sell: {unfilled_buy:>4}, {unfilled_sell:>4}, ' + \
                        f'long: {long_count:>4}, ' + \
                        f'\u2713 long: {profitable_long_count:>4}, ' + \
                        f'\u2713 long pct: {profitable_long_pct:>5.2f}%, ' + \
                        f'long profit pct: {mean_long_profit_pct:>6.4f}%, ' + \
                        f'occupancy rate: {occupancy_rate:>5.2f}%, ' # + \
                        # f'short: {short_count:>3}, ' + \
                        # f'\u2713 short: {profitable_short_count:>3}, ' + \
                        # f'\u2713 short pct: {profitable_short_count/short_count*100:>5.2f}%, ' + \
                        # f'short profit pct: {mean_short_profit_pct:>5.3f}%'  

                # account_growth = account_value/start_balance*100-100
                # stock_growth = price/start_price*100-100
                # pct_growth_diff = (account_growth+100)/(stock_growth+100)*100-100
                interval_per_trade = i/(long_count+short_count)
                ipt_since_last_plot = (i-prev_interval)/(long_count+short_count-prev_long_count-prev_short_count + 1)
                account_n_stock_str = \
                        f'Account Value: {account_value:>10.2f}, ' + \
                        f'accont growth: {account_growth:>6.2f}%, ' + \
                        f'stock growth: {stock_growth:>6.2f}%, ' +  \
                        f'pct growth diff: {pct_growth_diff:>6.2f}%, ' + \
                        f'interval per trade: {interval_per_trade:>4.2f}, ' + \
                        f'i/t since last plot: {ipt_since_last_plot:>4.2f}, ' #+ \
                    #   f'past 1000 interval growth: '

                print_n_log(complete_log_pth, decision_str, '\n', prediction_stat_str, '\n', account_n_stock_str)
                    
                    # print_n_log(complete_log_pth, f'ram: {psutil.virtual_memory().percent:.2f}%, vram: {torch.cuda.memory_allocated()/1024**3:.2f}GB')

            
            timers[4] += time.time() - st # time spent on printing
            st = time.time()

            if (i%draw_interval == 0) and (i != 0) and to_plot == True:
                
                # start_time_plot = time.time()
                plot(price_hist, account_value_hist, buy_decisions, sell_decisions, stock_growth, account_growth, ax1, ax2, annotations)

                # print_n_log(complete_log_pth, f'plotting completed in {time.time()-start_time_plot:.2f} seconds')
                
                prev_long_count = long_count
                prev_short_count = short_count
                prev_interval = i
            # print_n_log(complete_log_pth, account.evaluate())

            timers[5] += time.time() - st # time spent on plotting
            st = time.time()

            step_per_sec = step_per_sec * 0.9 + 0.1 * (1/(st-step_start))
            # print_n_log(complete_log_pth, st-step_start)

        time_analysis(timers)
        back_test_time = time.time()-start_time
        print_n_log(complete_log_pth, f'back test completed in {back_test_time:.2f} seconds')  

        plot(price_hist, account_value_hist, buy_decisions, sell_decisions, stock_growth, account_growth, ax1, ax2, annotations, show = to_plot)
        average_loss /= i

        global result_csv_pth, csv_header, model_pth, data_pth
        row_dict = {
            'symbol': None,
            'blocked_col_name': blocked_col_name,
            'account_value': account_value,
            'account_growth': account_growth,
            'stock_growth': stock_growth,
            'pct_growth_diff': pct_growth_diff,
            'interval_per_trade': interval_per_trade,
            'long_count': long_count,
            'profitable_long_count': profitable_long_count,
            'mean_long_profit_pct': mean_long_profit_pct,
            'occupancy_rate': occupancy_rate,
            'test_time': back_test_time,
            'model_pth': model_pth,
            'data_pth': data_pth,
        }
        # Check if the CSV file exists
        if not os.path.exists(result_csv_pth):
            # If not, create the file with the header
            with open(result_csv_pth, 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(csv_header)

        # Append the new row to the CSV file
        with open(result_csv_pth, 'a', newline='') as csvfile:
            csv_writer = csv.DictWriter(csvfile, fieldnames=csv_header)
            csv_writer.writerow(row_dict)
    except KeyboardInterrupt:
        account_growth = account_value/start_balance*100-100
        stock_growth = price/start_price*100-100
        # time_analysis(timers)
        back_test_time = time.time()-start_time
        print_n_log(complete_log_pth, f'back test completed in {back_test_time:.2f} seconds')  
        plot(price_hist, account_value_hist, buy_decisions, sell_decisions, stock_growth, account_growth, ax1, ax2, annotations, show = to_plot)
        
        raise KeyboardInterrupt
        
        

        # print_n_log(complete_log_pth, f'Epoch {epoch+1:3}/{num_epochs:3}, ' +
        #       f'Time per epoch: {(time.time()-start_time)/(epoch+1):.2f} seconds, ')
        
        


    

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

def main():
    global data_pth, model_pth, complete_log_pth, result_log_pth
    
    block_test = False
    to_print = True
    to_plot = True
    trade_data = True
    mock_trade = True

    weight_decay = 9.31

    if block_test:
        to_print = True
        to_plot = True


    should_log_result = False
    while True:
        tolog = input("Do you want to log the result? (y/n) ")
        if tolog == 'y':
            print_n_log(result_log_pth, f'\nBlock test: {block_test}')
            print_n_log(result_log_pth, f'\n{datetime.now()}Block test: {block_test}')
            purpose = input("What is the purpose of this back_test? ")
            print_n_log(result_log_pth, f'purpose: {purpose}')
            should_log_result = True
            break
        elif tolog == 'n':
            result_log_pth = None
            break
    
    
# dataname and model name are produced first, as they are used to make the log file paths.
    # data source
    start_time = time.time()
    time_str = '20200101_20200701'
    symbol = 'MSFT'
    data_type = '16feature0'
    data_name = f'bar_set_{time_str}_{symbol}_{data_type}_RAW'
    data_pth = f'data/csv/{data_name}.csv'
    #
    model = Seq2Seq(input_size, hidden_size, num_layers, output_size, prediction_window, dropout, device).to(device)
    model_name = f'last_model_{config_name}'
    model_pth = f'model/{model_name}.pt'
    model.load_state_dict(torch.load(model_pth))



    if trade_data:
        if mock_trade:
            mock_trade_data_pth = f'data/csv/mock_trade_{time_str}_{symbol}.csv'
            trade_df = pd.read_csv(mock_trade_data_pth)
        else:
            trade_data_pth = f'data/csv/trade_set_{time_str}_raw.csv'
            trade_df = pd.read_csv(trade_data_pth)
            # print_n_log(complete_log_pth, trade_df.head(3))
        trade_df['timestamp'] = pd.to_datetime(trade_df['timestamp'])
        trade_df['rounded_timestamp'] = trade_df['timestamp'].dt.round('1min')
        trade_df.set_index('rounded_timestamp', inplace=True)
        print(trade_df.head(5))
    else:
        trade_df = None
    # print_n_log(complete_log_pth, trade_df.head(3))


    
    block_str = 'noblock'
    if block_test:
        block_str = 'block'

    # log_pth and pic_pth_template
    complete_log_pth_template = f'log/complete_test_log/{data_name}--{model_name}_{{}}.txt'
    pic_pth_template = f'log/graph/{data_name}--{model_name}_{{i_th_attempt}}_block_{{blocked_col_name}}.png'

    i = 0
    while True:
        pic_pth_template_2 = pic_pth_template.format(i_th_attempt = i, blocked_col_name = '{blocked_col_name}')

        if not os.path.exists(complete_log_pth := complete_log_pth_template.format(i)): break
        i += 1

    # Weights
    arr = np.ones(prediction_window)
    for i in range(1, prediction_window):
        arr[i] = arr[i-1] * weight_decay
    weights = arr.reshape(1, prediction_window)
    
    #
    


    test_loader, col_names = \
        load_n_split_data(data_pth, 
                          hist_window, 
                          prediction_window, 
                          batch_size, 
                          train_ratio = 0, 
                          normalize = True,
                          test = True)

    # str logger

    block_str_lst = []
    test_strs_lst = []
    loss_lst = []
    with torch.no_grad():
        n = 0
        if block_test:
            n = feature_num

        for x in range(n+1):
            col_name = 'None'
            if not block_test or x == n: 
                x = None
            else:
                col_name = col_names[x]
            block_str = f'blocking column {x}:{col_name}'

            pic_pth = pic_pth_template_2.format(blocked_col_name = col_name)
            print_n_log(complete_log_pth, block_str)
            
            account = Account(initial_capital, ['MSFT'])
            # policy = RandomPolicy(account) # back ground has 0.023% of long profit pct... but it is only to do so in a bull market, and can only follow stock price; while the algorithm 
            policy = SimpleLongShort(account, buy_threshold=0.002 * pct_pred_multiplier, trade_data = trade_data)
            
            try:
                buy_decisions, sell_decisions, account_value_hist, price_hist, start_price, end_price, end_strs, loss= \
                    back_test(policy, model, test_loader, col_names, weights, trade_df, trade_data = trade_data, num_epochs = 1, blocked_col = x, blocked_col_name = col_name, to_plot = to_plot, to_print = to_print)
                
                plt.savefig(pic_pth)
                print_n_log(complete_log_pth, f'saving figure to {pic_pth}')
            except KeyboardInterrupt:
                plt.savefig(pic_pth)
                print_n_log(complete_log_pth, f'saving figure to {pic_pth}')
            
            # block_str_lst.append(block_str)
            # test_strs_lst.append(end_strs)
            # loss_lst.append(loss)

    print_n_log(complete_log_pth, f'Test completed in {time.time()-start_time:.2f} seconds')
    print_n_log(complete_log_pth, model_pth)
    if(should_log_result):
        purpose = input("Do you want to log the result?")
        print_n_log(result_log_pth, f'Result Interpretation: {purpose}')
    


if __name__ == "__main__":
    main()
    