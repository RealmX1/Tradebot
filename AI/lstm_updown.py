import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch # PyTorch
import torch.nn as nn # PyTorch neural network module
from torch.utils.data import Dataset, DataLoader # PyTorch data utilities
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import AdamW, SGD
# from apex.optimizers import FusedLAMB

import matplotlib.pyplot as plt
import os
import numpy as np
import time
import gc
import atexit
import copy
import random
import json
import csv
from datetime import datetime
import sys

sys.path.append('../')

# import custom files
from S2S import *
# from sim import *
from data_utils import * 
from model_structure_param import * # Define hyperparameters
from plot_util import *
from common import *

np.set_printoptions(precision=4, suppress=True) 
header = ['background_up', 'up_change_pred_pct', 'up_change_pred_precision', \
          'background_down', 'down_change_pred_pct', 'down_change_pred_precision', \
            'background_none', 'none_change_pred_pct', 'none_change_pred_precision', \
                'accuracy', 'accuracy_lst', \
                    'change_accuracy', 'change_accuracy_lst', \
                        'change_precision', 'change_direction_pred_precision_lst', \
                            'direction_precision', 'direction_precision_lst', \
                                'model_pth', 'time', 'best_k', 'epoch_num']



train_ratio     = 0.9
num_epochs      = 100

loss_fn = nn.MSELoss(reduction = 'none')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


torch.autograd.set_detect_anomaly(True)


def get_direction_diff(y_batch,y_pred):
    

    # start_time = time.time()
    # print('start get_direction_diff')
    # true_direction = y_batch-x_batch[:,-1,close_idx:close_idx+1]
    y_batch_below_threshold = np.zeros_like(y_batch, dtype=bool)
    y_batch_below_threshold[np.abs(y_batch) < policy_threshold] = True
    true_direction = np.clip(y_batch, 0, np.inf) # this turns negative to 0, positive to 1
    true_direction[true_direction != 0] = 1
    true_direction[true_direction == 0] = -1
    actual_direction = true_direction.copy() # SUPER STAR!
    true_direction[y_batch_below_threshold] = 0
    # true_direction = y_batch.cpu().numpy()

    y_pred_below_threshold = np.zeros_like(y_pred, dtype=bool)
    y_pred_below_threshold[np.abs(y_pred) < policy_threshold] = True
    pred_direction = np.clip(y_pred, 0, np.inf) # turn all 
    pred_direction[pred_direction != 0] = 1
    pred_direction[pred_direction == 0] = -1
    pred_direction[y_pred_below_threshold] = 0
    # pred_direction[pred_direction == 0.5] = 0
    # pred_direction = y_pred.clone().detach().cpu().numpy()

    # print('True: ', true_direction.shape)
    # print('Pred: ', pred_direction)

    instance_num =  true_direction.shape[0]
    prediction_min = true_direction.shape[1]

    all_cells = instance_num * prediction_min
    same_cells = np.count_nonzero(true_direction == pred_direction)

    all_cells_lst = np.full((prediction_min,), instance_num)
    same_cells_lst = np.count_nonzero(true_direction == pred_direction, axis = 0)

    all_change_lst = np.count_nonzero(true_direction != 0, axis = 0)
    true_change_pred_lst = np.count_nonzero((true_direction == pred_direction) & (true_direction != 0), axis = 0)
    all_change_direction_pred_lst = np.count_nonzero(pred_direction != 0, axis = 0)

    all_change = np.sum(all_change_lst)
    true_change_pred = np.sum(true_change_pred_lst)
    all_change_direction_pred = np.sum(all_change_direction_pred_lst)
    # print('all_cells: ',all_cells)
    # print('same_cells.shape: ',same_cells.shape)
    # print(type(true_direction))
    # print(true_direction.typedf())
    t_up = np.sum((true_direction == 1) & (pred_direction == 1))
    f_up = np.sum((true_direction != 1) & (pred_direction == 1))

    t_dn = np.sum((true_direction == -1) & (pred_direction == -1))
    f_dn = np.sum((true_direction != -1) & (pred_direction == -1))

    t_below_thres = np.sum((true_direction == 0) & (pred_direction == 0))
    f_below_thres = np.sum((true_direction != 0) & (pred_direction == 0))

    all_up = np.sum(true_direction == 1)
    all_down = np.sum(true_direction == -1)
    all_below_thres = np.sum(true_direction == 0)
    assert all_up + all_down + all_below_thres == all_cells
    assert t_up + f_up + t_dn + f_dn + t_below_thres + f_below_thres == all_cells
    assert same_cells == t_up + t_dn + t_below_thres, f'{same_cells} != {t_up} + {t_dn} + {t_below_thres}'

    # print('all_cells: ',all_cells)
    # print('all_cells_lst: ',all_cells_lst)
    # print('same_cells: ',same_cells)
    # print('same_cells_lst: ',same_cells_lst)
    # print('all_true: ',tp+tn)
    # print('all_false: ',fp+fn)
    # print('all_num: ', tp+tn+fp+fn)

    # print('get_direction_diff time: ', time.time()-start_time)
    all_true_dirction_change_pred_lst = np.sum((actual_direction == pred_direction) & (pred_direction != 0), axis = 0)
    true_change_direction_pred = np.sum(all_true_dirction_change_pred_lst)
    return all_cells, same_cells, \
            all_cells_lst, same_cells_lst, \
            \
            all_change, all_change_direction_pred, true_change_pred, true_change_direction_pred, \
            all_change_lst, all_change_direction_pred_lst, true_change_pred_lst, all_true_dirction_change_pred_lst, \
            \
            t_up, f_up, t_dn, f_dn, t_below_thres, f_below_thres, \
            all_up, all_down, all_below_thres


def calculate_policy_return(x_batch,y_batch,y_pred):
    pass

def count_tensor_num():
    num = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                num += 1
                # print(type(obj), obj.size())
        except:
            pass
    print('num of tensors: ', num)


train_weights = torch.pow(torch.tensor(weight_decay), torch.arange(prediction_window).float()).to(device)
# print("global weights: ",train_weights.shape)

def work(model, data_loader, optimizers, num_epochs = num_epochs, train = False, schedulers = None): # mode 0: train, mode 1: test, mode 2: PLOT
    try:
        if train:
            teacher_forcing_ratio = 0.0
            model.train()
        else:
            teacher_forcing_ratio = 0.0
            model.eval()
        start_time = time.time()
        same_cells = 0

        # count_tensor_num()


        all_prediction_lst             = np.zeros(prediction_window) # one elemnt for each minute of prediction window
        all_true_prediction_lst        = np.zeros(prediction_window)
        all_change_lst               = np.zeros(prediction_window)
        all_change_direction_pred_lst      = np.zeros(prediction_window)

        all_true_change_pred_lst = np.zeros(prediction_window)
        all_true_dirction_change_pred_lst = np.zeros(prediction_window)

        average_loss = 0

        inverse_mask = torch.linspace(1, 11, 10)
        # print ('inverse_mask.shape: ', inverse_mask.shape)
        global weights
        # print(weights)
        # ([1.0000, 0.8000, 0.6400, 0.5120, 0.4096, 0.3277, 0.2621, 0.2097, 0.1678,0.1342])
        # weights = torch.linspace(1, 0.1, steps=prediction_window)
        # ma_loss = None
        for epoch in range(num_epochs):
            epoch_loss = 0
            epoch_total_prediction = 0
            epoch_true_prediction = 0

            epoch_total_change = 0
            epoch_total_change_prediction = 0
            epoch_true_change_prediction = 0
            epoch_true_direction_prediction = 0
            # epoch_total_change = 0

            epoch_t_up = 0
            epoch_f_up = 0
            epoch_t_dn = 0
            epoch_f_dn = 0
            epoch_t_below_thres = 0
            epoch_f_below_thres = 0

            epoch_up = 0
            epoch_down = 0
            epoch_below_thres = 0
            i=0
            for i, (x_batch, y_batch, x_raw_close, timestamp) in enumerate(data_loader):

                # tensor_count = sum([len(gc.get_objects()) for obj in gc.get_objects() if isinstance(obj, torch.Tensor)])
                # print(f"There are currently {tensor_count} Torch tensors in memory.")
                # block_idx = [0,1,4,5,18,19] # it seems that open isn't useful at all.... since it is basically the last close price.
                # x_batch[:,:,block_idx] = 0
                
                # print('x_batch[0,:,:]: ', x_batch[0,:,:])
                # x_batch   [N, hist_window, feature_num]
                # y_batch & output  [N, prediction_window]
                x_batch = x_batch.float().to(device) # probably need to do numpy to pt tensor in the dataset; need testing on efficiency #!!! CAN'T BE DONE. before dataloarder they are numpy array, not torch tensor
                # print('one input: ', x_batch[0:,:,:])
                y_batch = y_batch.float().to(device)
                
                y_pred = model(x_batch, y_batch, teacher_forcing_ratio) # [N, prediction_window]
                # print('y_pred.shape: ', y_pred.shape)
                # print('y_batch.shape: ', y_pred.shape)
                # print('y_batch: ', y_batch[0,:])
                loss = loss_fn(y_pred, y_batch)
                # print('loss shape: ', loss.shape)

                loss_val = loss.clone().detach().cpu().mean().item()

                weighted_loss = loss * train_weights
                # print(weighted_loss.shape)
                # print(f"loss: {loss.shape}, weighted loss: {weighted_loss.shape}")
                final_loss = weighted_loss.mean()
                
                if train:
                    for optimizer in optimizers:
                        optimizer.zero_grad() # removing zero_grad doesn't improve training speed (unlike some claimed); need more testing
                    final_loss.backward()
                    for optimizer in optimizers:
                        optimizer.step()
                    if schedulers is not None:
                        for scheduler in schedulers:
                            scheduler.step(final_loss)
                memory_used = torch.cuda.memory_allocated()
                # print('memory_used: ', memory_used/1024/1024/8)

                # tmp = weighted_loss.detach()
                # if ma_loss is None:
                #     ma_loss = tmp.sum(axis = 0)
                # else:
                #     ma_loss *= 0.8
                #     ma_loss += 0.2*tmp.sum(axis = 0)
                y_batch_safe = y_batch.clone().detach().cpu().numpy()
                y_pred_safe = y_pred.clone().detach().cpu().numpy()

                all_cells, same_cells, \
                all_cells_lst, same_cells_lst, \
                all_change, all_change_direction_pred, true_change_pred, true_change_direction_pred, \
                acl, acpl, tcpl, cpdtl, \
                tp, fp, tn, fn, t_below_thres, f_below_thres, \
                all_up, all_down, all_below_thres= get_direction_diff(y_batch_safe, y_pred_safe)
                
                epoch_total_prediction += all_cells
                epoch_true_prediction += same_cells

                epoch_total_change += all_change
                epoch_total_change_prediction += all_change_direction_pred
                epoch_true_change_prediction += true_change_pred
                epoch_true_direction_prediction += true_change_direction_pred

                epoch_t_up += tp
                epoch_f_up += fp
                epoch_t_dn += tn
                epoch_f_dn += fn
                epoch_t_below_thres += t_below_thres
                epoch_f_below_thres += f_below_thres

                epoch_up += all_up
                epoch_down += all_down
                epoch_below_thres += all_below_thres


                all_prediction_lst += all_cells_lst
                all_true_prediction_lst += same_cells_lst

                all_change_lst += acl
                all_change_direction_pred_lst += acpl
                all_true_change_pred_lst += tcpl
                all_true_dirction_change_pred_lst += cpdtl

                
                epoch_loss += loss_val
                
            epoch_loss /= (i+1)
            average_loss += epoch_loss
            accuracy = epoch_true_prediction / epoch_total_prediction * 100
            change_pred_precision = epoch_true_change_prediction / epoch_total_change_prediction * 100
            change_pred_true_dir_precision = true_change_direction_pred / all_change_direction_pred * 100

            assert epoch_t_up + epoch_f_up + epoch_t_dn + epoch_f_dn + epoch_t_below_thres + epoch_f_below_thres == epoch_total_prediction
            assert epoch_up + epoch_down + epoch_below_thres == epoch_total_prediction
            # assert epoch_t_up + epoch_f_dn == epoch_up No longer applicable after adding below_thres
            # assert epoch_f_up + epoch_t_dn == epoch_down

            epoch_up_pred = epoch_t_up + epoch_f_up
            epoch_down_pred = epoch_f_dn + epoch_t_dn
            epoch_below_thres_pred = epoch_t_below_thres + epoch_f_below_thres

            time_per_epoch = (time.time()-start_time)/(epoch+1)
            
            general_stat_str=   f'Epoch {epoch+1:3}/{num_epochs:3}, ' + \
                            f'Loss: {                   epoch_loss:10.7f}, ' + \
                            f'Time/epoch: {             time_per_epoch:.2f} seconds, ' + \
                            f'\u2713 pred accuracy: {       accuracy:.2f}%, ' + \
                            f'\u2713 change pred precision: {change_pred_precision:.2f}%, ' + \
                            f'\u2713 direction precision: {change_pred_true_dir_precision:.2f}%, ' + \
                            f'Encocder LR: {            get_current_lr(optimizers[0]):9.8f},' # Decoder LR: {get_current_lr(optimizers[1]):9.8f}, '
            background_up = epoch_up / epoch_total_prediction * 100 
            up_change_pred_pct = epoch_up_pred / epoch_total_prediction * 100
            up_change_pred_precision = epoch_t_up / epoch_up_pred * 100

            change_up_str = f'Background \u2191: {  background_up:7.4f}%, ' + \
                            f'\u2191 Pred pct: {    up_change_pred_pct:7.4f}%, ' + \
                            f'\u2191 Precision: {   up_change_pred_precision:7.4f}%, '
            
            background_down = epoch_down / epoch_total_prediction * 100
            down_change_pred_pct = epoch_down_pred / epoch_total_prediction * 100
            down_change_pred_precision = epoch_t_dn / epoch_down_pred * 100

            change_down_str=f'Background \u2193: {  background_down:7.4f}%, ' + \
                            f'\u2193 Pred pct: {    down_change_pred_pct:7.4f}%, ' + \
                            f'\u2193 Precision: {   down_change_pred_precision:7.4f}%, '

            background_none = epoch_below_thres / epoch_total_prediction * 100
            none_change_pred_pct = epoch_below_thres_pred / epoch_total_prediction * 100
            none_change_pred_precision = epoch_t_below_thres / epoch_below_thres_pred * 100
                
            change_none_str=f'Background \u2192: {  background_none:7.4f}%, ' + \
                            f'\u2192 Pred pct: {    none_change_pred_pct:7.4f}%, ' + \
                            f'\u2192 Precision: {   none_change_pred_precision:7.4f}%, '
            
            
            background_up, up_change_pred_pct, up_change_pred_precision, background_down, down_change_pred_pct, down_change_pred_precision, background_none, none_change_pred_pct, none_change_pred_precision
            print(general_stat_str)
            print(change_up_str)
            print(change_down_str)
            print(change_none_str)
                # f'Weighted Loss: {final_loss.item():10.7f}, MA Loss: {ma_loss.mean().item():10.7f}') 
                
            del x_batch, y_batch, y_pred, loss, weighted_loss, final_loss
        
        average_loss /= num_epochs
        accuracy_lst = all_true_prediction_lst / all_prediction_lst * 100
        accuracy_lst_print = [round(x, 3) for x in accuracy_lst]
        change_accuracy_lst = all_true_change_pred_lst / all_change_lst * 100
        change_accuracy_lst_print = [round(x, 3) for x in change_accuracy_lst]

        change_direction_pred_precision_lst = all_true_change_pred_lst / all_change_direction_pred_lst * 100
        change_direction_pred_precision_lst_print = [round(x, 3) for x in change_direction_pred_precision_lst]
        true_change_direction_pred_precision_lst = all_true_dirction_change_pred_lst/ all_change_direction_pred_lst * 100
        true_change_direction_pred_precision_lst_print = [round(x, 3) for x in true_change_direction_pred_precision_lst]
        print('Accuracy List: ', accuracy_lst_print)
        print('Change Accuracy List: ', change_accuracy_lst_print)

        print('Change Direction Pred Precision List: ', change_direction_pred_precision_lst_print)
        print('True Change Direction Pred Precision List: ', true_change_direction_pred_precision_lst_print)
        # print(f'completed in {time.time()-start_time:.2f} seconds')



        if train:
            return average_loss
        else:
            change_accuracy = epoch_true_change_prediction / epoch_total_change * 100
            change_precision = sum(change_direction_pred_precision_lst)/len(change_direction_pred_precision_lst)

            direction_precision = sum(true_change_direction_pred_precision_lst) / len(true_change_direction_pred_precision_lst)
            direction_precision_lst = true_change_direction_pred_precision_lst
            # Define a sample row as a dictionary
            row_dict = {
                'background_up': background_up,
                'up_change_pred_pct': up_change_pred_pct,
                'up_change_pred_precision': up_change_pred_precision,
                'background_down': background_down,
                'down_change_pred_pct': down_change_pred_pct,
                'down_change_pred_precision': down_change_pred_precision,
                'background_none': background_none,
                'none_change_pred_pct': none_change_pred_pct,
                'none_change_pred_precision': none_change_pred_precision,
                'accuracy': accuracy,
                'accuracy_lst': accuracy_lst,
                'change_accuracy': change_accuracy,
                'change_accuracy_lst': change_accuracy_lst,
                'change_precision': change_precision,
                'change_direction_pred_precision_lst': change_direction_pred_precision_lst,
                'direction_precision': direction_precision,
                'direction_precision_lst': direction_precision_lst
            }
            print(row_dict['accuracy'])
            print(row_dict['change_accuracy'])
            print(row_dict['change_precision'])
            print(row_dict['direction_precision'])

            # print(row_dict)
            return true_change_direction_pred_precision_lst, average_loss, row_dict, 
    except KeyboardInterrupt:
        average_loss /= num_epochs
        accuracy_lst = all_true_prediction_lst / all_prediction_lst * 100
        accuracy_lst_print = [round(x, 3) for x in accuracy_lst]
        change_accuracy_lst = all_true_change_pred_lst / all_change_lst * 100
        change_accuracy_lst_print = [round(x, 3) for x in change_accuracy_lst]

        change_direction_pred_precision_lst = all_true_change_pred_lst / all_change_direction_pred_lst * 100
        change_direction_pred_precision_lst_print = [round(x, 3) for x in change_direction_pred_precision_lst]
        true_change_direction_pred_precision_lst = all_true_dirction_change_pred_lst/ all_change_direction_pred_lst * 100
        true_change_direction_pred_precision_lst_print = [round(x, 3) for x in true_change_direction_pred_precision_lst]
        print('Accuracy List: ', accuracy_lst_print)
        print('Change Accuracy List: ', change_accuracy_lst_print)
        print('Change Precision List: ', change_direction_pred_precision_lst_print)
        print('Prediction of Change Precision List: ', true_change_direction_pred_precision_lst_print)
        raise KeyboardInterrupt

def plot(predictions, targets, test_size):
    # Plot the results
    print('total entry: ',predictions.shape[0])
    x = np.arange(len(predictions))
    print('predictions.shape: ',predictions.shape)
    plt.plot(targets, label='Actual')
    plt.plot(predictions, label='Predicted',linestyle='dotted')
    plt.legend()
    plt.axvline(x=test_size, color='r')
    plt.show()

def get_current_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def save_params(best_prediction, optimizers, model_state, last_model_pth, best_model_state, best_model_pth, model_training_param_path, has_improvement, best_k, epoch_num):
    print('saving params...')

    encoder_lr = get_current_lr(optimizers[0])
    decoder_lr = get_current_lr(optimizers[1])
    with open(model_training_param_path, 'w') as f:
        json.dump({'encoder_learning_rate': encoder_lr, 'decoder_learning_rate': decoder_lr, 'best_prediction': best_prediction, 'best_k':best_k, 'epoch_num':epoch_num}, f)
    print('saving model to: ', last_model_pth)
    torch.save(model_state, last_model_pth)
    if has_improvement:
        print('saving best model to: ', best_model_pth)
        torch.save(best_model_state, best_model_pth)
    print('done.')
    
def moving_average(data, window_size = 5):
    weights = np.repeat(1.0, window_size) / window_size
    return np.convolve(data, weights, mode='valid')

def main():
    train_purporse_log_pth = '../log/train_purpose_log.txt'
    should_log_result = False
    while True:
        tolog = input("Do you want to log the result? (y/n) ")
        if tolog == 'y':
            print_n_log(train_purporse_log_pth, f'{datetime.now()}')
            purpose = input("What is the purpose of this back_test? ")
            print_n_log(train_purporse_log_pth, f'purpose: {purpose}')
            should_log_result = True
            break
        elif tolog == 'n':
            break
    # CHANGE CONFIG NAME to save a new model
    start_time = time.time()
    print('loading data & model')
    # data_pth = 'data/cdl_test_2.csv'
    data_pth = training_data_path
    best_model_pth = f'../model/model_{config_name}.pt'
    last_model_pth = f'../model/last_model_{config_name}.pt'
    model_training_param_path = f'../model/training_param_{config_name}.json'

    csv_file_path = "../log/training_log.csv"
    print('loaded in ', time.time()-start_time, ' seconds')
    
    train_loader, test_loader = load_n_split_data(data_pth, hist_window, prediction_window, batch_size, train_ratio)
    


    print('loading model')
    start_time = time.time()
    model = Seq2Seq(input_size, hidden_size, num_layers, output_size, prediction_window, dropout, device, attention = True).to(device)
    best_model = copy.deepcopy(model)
    if os.path.exists(last_model_pth):
        print('Loading existing model')
        model.load_state_dict(torch.load(last_model_pth))
        best_model.load_state_dict(torch.load(best_model_pth))
        with open(model_training_param_path, 'r') as f:
            saved_data = json.load(f)
            encoder_lr = saved_data['encoder_learning_rate']
            decoder_lr = saved_data['decoder_learning_rate']
            best_prediction = saved_data['best_prediction']
            best_k = saved_data['best_k']
            epoch_num = saved_data['epoch_num']
            print('best_k: ', best_k)
            start_best_prediction = best_prediction
    else:
        print('No existing model')
        encoder_lr = learning_rate
        decoder_lr = learning_rate
        best_prediction = 0.0
        best_k = 0.0
        epoch_num = 0
        start_best_prediction = 0.0
    best_model_state = best_model.state_dict()
   
    print(model)
    print(f'model loading completed in {time.time()-start_time:.2f} seconds')


    # optimizer = SGD(model.parameters(), lr=learning_rate)
    encoder_optimizer = AdamW(model.encoder.parameters(),weight_decay=1e-5, lr=encoder_lr)
    decoder_optimizer = AdamW(model.decoder.parameters(),weight_decay=1e-5, lr=decoder_lr)
    encoder_scheduler = ReduceLROnPlateau(encoder_optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience, threshold=scheduler_threshold)
    decoder_scheduler = ReduceLROnPlateau(decoder_optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience, threshold=scheduler_threshold)

    optimizers = [encoder_optimizer, decoder_optimizer]
    schedulers = [encoder_scheduler, decoder_scheduler]
    

    

    try:
        plt.ion
        # Train the model
        start_time = time.time()
        print('Training model')
        test_every_x_epoch = 1
        # row_dict = {
        #         'background_up': background_up,
        #         'up_change_pred_pct': up_change_pred_pct,
        #         'up_change_pred_precision': up_change_pred_precision,
        #         'background_down': background_down,
        #         'down_change_pred_pct': down_change_pred_pct,
        #         'down_change_pred_precision': down_change_pred_precision,
        #         'background_none': background_none,
        #         'none_change_pred_pct': none_change_pred_pct,
        #         'none_change_pred_precision': none_change_pred_precision,
        #         'accuracy': accuracy,
        #         'accuracy_lst': accuracy_lst,
        #         'change_accuracy': change_accuracy,
        #         'change_accuracy_lst': change_accuracy_lst,
        #         'change_precision': change_precision,
        #         'change_direction_pred_precision_lst': change_direction_pred_precision_lst,
        #         'direction_precision': direction_precision,
        #         'direction_precision_lst': direction_precision_lst
        #     }
        plot_hist_dict = {}
        fig, (ax_0, ax_1, ax_2) = plt.subplots(3,1, figsize=(10,10))
        fig.subplots_adjust(top=0.95, bottom=0.05)
        manager = plt.get_current_fig_manager()
        manager.full_screen_toggle()
        # test_change_precision_hist = np.zeros((prediction_window,1))
        # eval_loss_hist = []
        # train_loss_hist = []

        weights_lst = [] # weights to test out.
        for k in range(-10,11):
            arr = np.ones(prediction_window)
            w_d = 0.0
            for i in range(1, prediction_window):
                w_d = pow(0.8,k) # !!!!!!!!!!!!!!!!!!!!!!!! use best k to calcualte best weight decay.
                arr[i] = arr[i-1] * w_d
            weights = arr.reshape(1,prediction_window)
            weights_lst.append(weights)

        has_improvement = False
        
        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}/{num_epochs}')
            if test_every_x_epoch != 0 and epoch % test_every_x_epoch == 0:
                with torch.no_grad():
                    true_change_direction_pred_precision_lst, average_loss, row_dict = work(model, test_loader, optimizers, num_epochs = 1, train = False)
                    for key, value in row_dict.items():
                        if key not in plot_hist_dict:
                            plot_hist_dict[key] = []
                        plot_hist_dict[key].append(value)
                    plot_history(plot_hist_dict, ax_0, ax_1, ax_2)
                    
                    row_dict['model_pth'] = last_model_pth
                    row_dict['time'] = datetime.now()
                    row_dict['best_k'] = best_k
                    row_dict['epoch_num'] = epoch_num
                    if not os.path.exists(csv_file_path):
                        # If not, create the file with the header
                        with open(csv_file_path, 'w', newline='') as csvfile:
                            csv_writer = csv.DictWriter(csvfile, fieldnames=header)
                            csv_writer.writeheader()

                    # Append the new row to the CSV file
                    with open(csv_file_path, 'a', newline='') as csvfile:
                        csv_writer = csv.DictWriter(csvfile, fieldnames=header)
                        csv_writer.writerow(row_dict)


                    best_weights = weights_lst[0]
                    highest_change_precision = 0.0
                    improved = False
                    for k, weights in enumerate(weights_lst):
                        # print(weights)
                        change_precision = true_change_direction_pred_precision_lst.reshape(1,prediction_window)*weights
                        change_precision = change_precision.sum()/np.sum(weights)
                        if change_precision > highest_change_precision:
                            highest_change_precision = change_precision
                            best_weights = weights
                        if change_precision > best_prediction: 
                            best_k = k
                            has_improvement = True
                            improved = True
                            print(weights)
                            print(f'\nNEW BEST prediction: {change_precision:.4f}% at k: {best_k}\n')
                            best_prediction = change_precision
                            best_model_state = model.state_dict()
                    if not improved:
                        print(best_weights)
                        print(f'\ncurrent best change prediction precision: {highest_change_precision:.4f}% at k: {best_k}\n')


                    '''
                    PLOT true_change_direction_pred_precision_lst
                    if epoch == 0:
                        test_change_precision_hist[:,0] = true_change_direction_pred_precision_lst
                    else:
                        test_change_precision_hist = np.concatenate((test_change_precision_hist, true_change_direction_pred_precision_lst.reshape(prediction_window,1)), axis=1)
                    eval_loss_hist.append(average_loss)


                    plt.clf()
                    
                    for i in range(prediction_window):
                        minute_i_precision = test_change_precision_hist[i,:]
                        plt.plot(minute_i_precision, label=f'{i+1} min precision', linestyle='solid')
                    plt.plot(test_change_precision_hist.mean(axis=0), label=f'average precision', linestyle='dashed')
                    weighted_precision_lst = np.matmul(best_weights,test_change_precision_hist)/np.sum(best_weights)
                    # print('weighted_precision_lst', weighted_precision_lst.shape) # shape (1, epoch)
                    plt.plot(weighted_precision_lst[0], label=f'weighted precision', linestyle='dotted')
                    '''
                # plt.clf()
                # plt.plot(moving_average(eval_loss_hist, 3), label=f'loss', linestyle='solid')

                # actually train the model
            average_loss = work(model, train_loader, optimizers, test_every_x_epoch, train = True, schedulers = schedulers)
            epoch_num += 1
                # train_loss_hist.append(average_loss)
                # plt.plot(train_loss_hist, label=f'train loss', linestyle='dotted')
            # if epoch == 0:
            # plt.legend() 
            # plt.pause(0.5)
        print(f'training completed in {time.time()-start_time:.2f} seconds')
        
        print('\n\n')
        if best_prediction > start_best_prediction:
            print(f'improved from {start_best_prediction:.2f}% to {best_prediction:.2f}%')
        else:
            print(f'NO IMPROVEMENET from {start_best_prediction:.2f}%')
        print('\n\n')

        print(test_change_precision_hist.shape)
        # predictions = np.mean(predictions, axis = )
        # plt.ion()
        # for i in range(prediction_window):
        #     predictions = test_change_precision_hist[i,:]
        #     plt.plot(predictions, label=f'{i+1} min prediction', linestyle='solid')
        # plt.plot(test_change_precision_hist.mean(axis=0), label=f'average prediction', linestyle='dashed')
        # weights = np.linspace(1, 0.1, num=prediction_window)
        # weights = weights.reshape(prediction_window,1)
        # plt.plot((test_change_precision_hist*weights).sum(axis=0)/np.sum(weights), label=f'weighted prediction', linestyle='dotted')
        # plt.legend()

        print('Training Complete')
        # plt.show()
        # plt.clf()
        # plt.ioff()

        encoder_lr = get_current_lr(encoder_optimizer)
        decoder_lr = get_current_lr(decoder_optimizer)
        lrs = [encoder_lr, decoder_lr]

        # Test the model
        # start_time = time.time()
        # print('Testing model')
        # with torch.no_grad():
        #     test_change_direction_pred_precision_lst, average_loss = work(model, test_loader, optimizers, num_epochs = 1, mode = 1)
        #     test_change_precision_hist[:,0] = test_change_direction_pred_precision_lst
        #     plt.clf()
        #     for i in range(prediction_window):
        #         predictions = test_change_precision_hist[i,:]
        #         plt.plot(predictions, 0, label=f'{i+1} min accuracy', marker = 'o')
        #     plt.plot(test_change_precision_hist.mean(axis=0), 0, label=f'average accuracy', marker = 'o')
        #     plt.plot((test_change_precision_hist*weights).sum(axis=0)/np.sum(weights), label=f'weighted accuracy', linestyle='dotted')
        #     plt.legend()
        #     plt.ylim(0, 1)
        #     plt.show()
        #     plt.pause(1)
        # print(f'testing completed in {time.time()-start_time:.2f} seconds')
        last_model_state = model.state_dict()
        save_params(best_prediction, optimizers, last_model_state, last_model_pth, best_model_state, best_model_pth, model_training_param_path, has_improvement, best_k, epoch_num) 
        print('Normal exit. Model saved.')
        torch.cuda.empty_cache()
        gc.collect()
    except KeyboardInterrupt or Exception or TypeError:
        # save the model if the training was interrupted by keyboard input
        last_model_state = model.state_dict()
        save_params(best_prediction, optimizers, last_model_state, last_model_pth, best_model_state, best_model_pth, model_training_param_path, has_improvement, best_k, epoch_num)
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == '__main__':
    main()
    # cProfile.run('main()') # this shows execution time of each function. Might be useful for debugging & accelerating in detail.