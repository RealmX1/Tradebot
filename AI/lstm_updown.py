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
from transformer import *

np.set_printoptions(precision=4, suppress=True) 
header = ['background_up', 'up_change_pred_pct', 'up_change_pred_precision', \
          'background_dn', 'dn_change_pred_pct', 'dn_change_pred_precision', \
            'background_none', 'none_change_pred_pct', 'none_change_pred_precision', \
                'accuracy', 'accuracy_lst', \
                    'pred_thres_change_accuracy', 'pred_thres_change_accuracy_lst', \
                        'pred_thres_change_precision', 'pred_thres_change_percision_lst', \
                            'pred_thres_actual_change_precision', 'pred_thres_actual_change_precision_lst', 'pred_thres_up_actual_precision', 'pred_thres_dn_actual_precision',\
                                'model_pth', 'time', 'best_k', 'epoch_num']


loss_fn = nn.MSELoss(reduction = 'none')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


torch.autograd.set_detect_anomaly(True)


def get_direct_diff(y_batch,y_pred):
    y_batch_below_threshold = np.zeros_like(y_batch, dtype=bool)
    y_batch_below_threshold[np.abs(y_batch) < policy_threshold] = True
    actual_direct = np.clip(y_batch, 0, np.inf) # this turns negative to 0
    actual_direct[actual_direct != 0] = 1
    actual_direct[actual_direct == 0] = -1 # turns positive to 1
    actual_thres_direct = actual_direct.copy()
    actual_thres_direct[y_batch_below_threshold] = 0

    y_pred_below_threshold = np.zeros_like(y_pred, dtype=bool)
    y_pred_below_threshold[np.abs(y_pred) < policy_threshold] = True
    pred_direct = np.clip(y_pred, 0, np.inf) # turn all 
    pred_direct[pred_direct != 0] = 1
    pred_direct[pred_direct == 0] = -1
    pred_thres_direct = pred_direct.copy()
    pred_thres_direct[y_pred_below_threshold] = 0



    batch_size  = y_batch.shape[0]
    pred_window = y_batch.shape[1]

    all_cells_lst = np.full((pred_window,), batch_size)
    all_cells = batch_size * pred_window

    same_thres_cells_lst = np.count_nonzero(actual_thres_direct == pred_thres_direct, axis = 0)
    same_thres_cells = np.count_nonzero(actual_thres_direct == pred_thres_direct)

    actual_thres_change_lst = np.count_nonzero(actual_thres_direct != 0, axis = 0)
    true_pred_thres_change_lst = np.count_nonzero((actual_thres_direct == pred_thres_direct) & (actual_thres_direct != 0), axis = 0)
    all_pred_thres_change_lst = np.count_nonzero(pred_thres_direct != 0, axis = 0)

    actual_thres_change = np.sum(actual_thres_change_lst)
    true_pred_thres_change = np.sum(true_pred_thres_change_lst)
    all_pred_thres_change = np.sum(all_pred_thres_change_lst)
    
    t_thres_up = np.sum((actual_thres_direct == 1) & (pred_thres_direct == 1))
    f_thres_up = np.sum((actual_thres_direct != 1) & (pred_thres_direct == 1))

    t_thres_dn = np.sum((actual_thres_direct == -1) & (pred_thres_direct == -1))
    f_thres_dn = np.sum((actual_thres_direct != -1) & (pred_thres_direct == -1))

    t_thres_no = np.sum((actual_thres_direct == 0) & (pred_thres_direct == 0))
    f_thres_no = np.sum((actual_thres_direct != 0) & (pred_thres_direct == 0))

    actual_thres_up = np.sum(actual_thres_direct == 1)
    actual_thres_dn = np.sum(actual_thres_direct == -1)
    actual_thres_no = np.sum(actual_thres_direct == 0)

    assert actual_thres_up + actual_thres_dn + actual_thres_no == all_cells
    assert t_thres_up + f_thres_up + t_thres_dn + f_thres_dn + t_thres_no + f_thres_no == all_cells
    assert same_thres_cells == t_thres_up + t_thres_dn + t_thres_no, f'{same_thres_cells} != {t_thres_up} + {t_thres_dn} + {t_thres_no}'



    pred_thres_up_actual_up_lst = np.sum((actual_direct == 1) & (pred_thres_direct == 1), axis = 0)
    pred_thres_dn_actual_dn_lst = np.sum((actual_direct == -1) & (pred_thres_direct == -1), axis = 0)
    pred_thres_up_actual_up = np.sum(pred_thres_up_actual_up_lst)
    pred_thres_dn_actual_dn = np.sum(pred_thres_dn_actual_dn_lst)

    true_pred_thres_actual_change_lst = pred_thres_up_actual_up_lst + pred_thres_dn_actual_dn_lst
    true_pred_thres_actual_change = np.sum(true_pred_thres_actual_change_lst)

    pred_thres_up = np.sum(pred_thres_direct == 1)

    # print('get_direct_diff time: ', time.time()-start_time)

    return all_cells, same_thres_cells, \
            all_cells_lst, same_thres_cells_lst, \
            \
            actual_thres_up, actual_thres_dn, actual_thres_no, \
            t_thres_up, f_thres_up, t_thres_dn, f_thres_dn, t_thres_no, f_thres_no, \
            \
            actual_thres_change, all_pred_thres_change, true_pred_thres_change, true_pred_thres_actual_change, pred_thres_up_actual_up, pred_thres_dn_actual_dn, pred_thres_up,\
            actual_thres_change_lst, all_pred_thres_change_lst, true_pred_thres_change_lst, true_pred_thres_actual_change_lst

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
        same_thres_cells = 0

        # count_tensor_num()


        all_prediction_lst             = np.zeros(prediction_window) # one elemnt for each minute of prediction window
        all_true_prediction_lst        = np.zeros(prediction_window)
        actual_thres_change_lst               = np.zeros(prediction_window)
        all_pred_thres_change_lst      = np.zeros(prediction_window)

        all_true_pred_thres_change_lst = np.zeros(prediction_window)
        all_true_pred_actual_thres_change_lst = np.zeros(prediction_window)

        average_loss = 0

        inverse_mask = torch.linspace(1, 11, 10)
        # print ('inverse_mask.shape: ', inverse_mask.shape)
        # print(weights)
        # ([1.0000, 0.8000, 0.6400, 0.5120, 0.4096, 0.3277, 0.2621, 0.2097, 0.1678,0.1342])
        # weights = torch.linspace(1, 0.1, steps=prediction_window)
        # ma_loss = None
        for epoch in range(num_epochs):
            epoch_loss = 0
            epoch_total_prediction = 0
            epoch_true_pred_thres = 0

            epoch_total_change = 0
            epoch_total_change_prediction = 0
            epoch_true_pred_thres_change = 0
            epoch_actual_thres_direct_prediction = 0
            # New
            epoch_pred_thres_up_actual_up = 0
            epoch_pred_thres_dn_actual_dn = 0
            epoch_pred_thres_up = 0
            epoch_pred_thres_dn = 0

            epoch_t_thres_up = 0
            epoch_f_thres_up = 0
            epoch_t_thres_dn = 0
            epoch_f_thres_dn = 0
            epoch_t_thres_no = 0
            epoch_f_thres_no = 0

            epoch_up = 0
            epoch_dn = 0
            epoch_below_thres = 0
            i=0
            for i, (x_batch, y_batch, x_raw_close, timestamp) in enumerate(data_loader):

                # tensor_count = sum([len(gc.get_objects()) for obj in gc.get_objects() if isinstance(obj, torch.Tensor)])
                # print(f"There are currently {tensor_count} Torch tensors in memory.")
                # block_idx = [0,1,4,5,18,19] # it seems that open isn't useful at all.... since it is basically the last close price.
                # x_batch[:,:,block_idx] = 0
                
                # print('x_batch[0,:,:]: ', x_batch[0,:,:])
                # x_batch   [N, hist_window, feature_num]
                # y_batch & out_thres_uput  [N, prediction_window]
                x_batch = x_batch.float().to(device) # probably need to do numpy to pt tensor in the dataset; need testing on efficiency #!!! CAN'T BE DONE. before dataloarder they are numpy array, not torch tensor
                # print('one input: ', x_batch[0:,:,:])
                y_batch = y_batch.float().to(device)
                
                if model_type == 'transformer':
                    if train:
                        y_pred = model.predict(x_batch, y_batch)
                    else:
                        y_batch_None = torch.zeros_like(y_batch).to(device)
                        y_pred = model.predict(x_batch, y_batch_None)
                elif model_type == 'lstm':
                    y_pred = model(x_batch, y_batch, teacher_forcing_ratio) # [N, prediction_window]
                # print('y_pred.shape: ', y_pred.shape)
                # print('y_batch.shape: ', y_pred.shape)
                # print('y_batch: ', y_batch[0,:])
                loss = loss_fn(y_pred, y_batch) # 
                # print('loss shape: ', loss.shape)

                loss_val = loss.clone().detach().cpu().mean().item()

                weighted_loss = loss * train_weights
                 # can try do more crazy loss weights -- for example; direct precision!
                # print("train_weights: ", train_weights)
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

                y_batch_safe = y_batch.clone().detach().cpu().numpy()[:,:,0]

                y_pred_safe = y_pred.clone().detach().cpu().numpy()[:,:,0]

                all_cells, same_thres_cells, \
                all_cells_lst, same_thres_cells_lst, \
                \
                actual_thres_up, actual_thres_dn, actual_thres_no,\
                t_thres_up, f_thres_up, t_thres_dn, f_thres_dn, t_thres_no, f_thres_no, \
                \
                actual_thres_change, all_pred_thres_change, true_pred_thres_change, true_pred_thres_actual_change, pred_thres_up_actual_up, pred_thres_dn_actual_dn, pred_thres_up, \
                atcl, aptcl, tptcl, tptacl = get_direct_diff(y_batch_safe, y_pred_safe)
                # print("DEBUG!!!!!")
                # for x in tmp:
                #     print(x)
                
                epoch_total_prediction += all_cells
                epoch_true_pred_thres += same_thres_cells

                epoch_total_change += actual_thres_change
                epoch_total_change_prediction += all_pred_thres_change
                epoch_true_pred_thres_change += true_pred_thres_change
                epoch_actual_thres_direct_prediction += true_pred_thres_actual_change
                # News
                epoch_pred_thres_up += pred_thres_up
                epoch_pred_thres_dn += all_pred_thres_change - pred_thres_up
                epoch_pred_thres_up_actual_up += pred_thres_up_actual_up
                epoch_pred_thres_dn_actual_dn += pred_thres_dn_actual_dn


                epoch_t_thres_up += t_thres_up
                epoch_f_thres_up += f_thres_up
                epoch_t_thres_dn += t_thres_dn
                epoch_f_thres_dn += f_thres_dn
                epoch_t_thres_no += t_thres_no
                epoch_f_thres_no += f_thres_no

                epoch_up += actual_thres_up
                epoch_dn += actual_thres_dn
                epoch_below_thres += actual_thres_no


                all_prediction_lst += all_cells_lst
                all_true_prediction_lst += same_thres_cells_lst

                actual_thres_change_lst += atcl
                all_pred_thres_change_lst += aptcl
                all_true_pred_thres_change_lst += tptcl
                all_true_pred_actual_thres_change_lst += tptacl

                
                epoch_loss += loss_val
                
            epoch_loss /= (i+1)
            average_loss += epoch_loss
            accuracy = epoch_true_pred_thres / epoch_total_prediction * 100
            change_pred_precision = epoch_true_pred_thres_change / epoch_total_change_prediction * 100
            pred_thres_change_actual_pred_thres_change_precision = (epoch_pred_thres_up_actual_up+epoch_pred_thres_dn_actual_dn) / (epoch_total_change_prediction) * 100
            pred_thres_up_actual_precision = epoch_pred_thres_up_actual_up / epoch_pred_thres_up * 100
            pred_thres_dn_actual_precision = epoch_pred_thres_dn_actual_dn / epoch_pred_thres_dn * 100
            

            assert epoch_t_thres_up + epoch_f_thres_up + epoch_t_thres_dn + epoch_f_thres_dn + epoch_t_thres_no + epoch_f_thres_no == epoch_total_prediction
            assert epoch_up + epoch_dn + epoch_below_thres == epoch_total_prediction
            # assert epoch_t_thres_up + epoch_f_dn == epoch_up No longer applicable after adding below_thres
            # assert epoch_f_thres_up + epoch_t_dn == epoch_dn

            epoch_up_pred = epoch_t_thres_up + epoch_f_thres_up
            epoch_dn_pred = epoch_f_thres_dn + epoch_t_thres_dn
            epoch_below_thres_pred = epoch_t_thres_no + epoch_f_thres_no

            time_per_epoch = (time.time()-start_time)/(epoch+1)
            
            # print('debug: ', pred_thres_change_actual_pred_thres_change_precision)
            general_stat_str=   f'Epoch {epoch+1:3}/{num_epochs:3}, ' + \
                            f'Loss: {                   epoch_loss:10.7f}, \n' + \
                            f'Time/epoch: {             time_per_epoch:.2f} seconds, ' + \
                            f'\u2713 pred accuracy: {       accuracy:.2f}%, ' + \
                            f'\u2713 change pred precision: {change_pred_precision:.2f}%, \n' + \
                            f'\u2713 pred_thres_change_actual precision: {pred_thres_change_actual_pred_thres_change_precision:.2f}%, ' + \
                            f'\u2713 pred_thres_up_actual precision: {pred_thres_up_actual_precision:.2f}%, ' + \
                            f'\u2713 pred_thres_dn_actual precision: {pred_thres_dn_actual_precision:.2f}%, ' + \
                            f'Encocder LR: {            get_current_lr(optimizers[0]):9.8f},' # Decoder LR: {get_current_lr(optimizers[1]):9.8f}, '
            background_up = epoch_up / epoch_total_prediction * 100 
            up_change_pred_pct = epoch_up_pred / epoch_total_prediction * 100
            up_change_pred_precision = epoch_t_thres_up / epoch_up_pred * 100

            change_up_str = f'Background \u2191: {  background_up:7.4f}%, ' + \
                            f'\u2191 Pred pct: {    up_change_pred_pct:7.4f}%, ' + \
                            f'\u2191 Precision: {   up_change_pred_precision:7.4f}%, '
            
            background_dn = epoch_dn / epoch_total_prediction * 100
            dn_change_pred_pct = epoch_dn_pred / epoch_total_prediction * 100
            dn_change_pred_precision = epoch_t_thres_dn / epoch_dn_pred * 100

            change_dn_str=f'Background \u2193: {  background_dn:7.4f}%, ' + \
                            f'\u2193 Pred pct: {    dn_change_pred_pct:7.4f}%, ' + \
                            f'\u2193 Precision: {   dn_change_pred_precision:7.4f}%, '

            background_none = epoch_below_thres / epoch_total_prediction * 100
            none_change_pred_pct = epoch_below_thres_pred / epoch_total_prediction * 100
            none_change_pred_precision = epoch_t_thres_no / epoch_below_thres_pred * 100
                
            change_none_str=f'Background \u2192: {  background_none:7.4f}%, ' + \
                            f'\u2192 Pred pct: {    none_change_pred_pct:7.4f}%, ' + \
                            f'\u2192 Precision: {   none_change_pred_precision:7.4f}%, '
            
            
            background_up, up_change_pred_pct, up_change_pred_precision, background_dn, dn_change_pred_pct, dn_change_pred_precision, background_none, none_change_pred_pct, none_change_pred_precision
            print(general_stat_str)
            print(change_up_str)
            print(change_dn_str)
            print(change_none_str)
                # f'Weighted Loss: {final_loss.item():10.7f}, MA Loss: {ma_loss.mean().item():10.7f}') 
                
            del x_batch, y_batch, y_pred, loss, weighted_loss, final_loss
        
        average_loss /= num_epochs
        accuracy_lst = all_true_prediction_lst / all_prediction_lst * 100
        accuracy_lst_print = [round(x, 3) for x in accuracy_lst]
        pred_thres_change_accuracy_lst = all_true_pred_thres_change_lst / actual_thres_change_lst * 100
        pred_thres_change_accuracy_lst_print = [round(x, 3) for x in pred_thres_change_accuracy_lst]

        pred_thres_change_percision_lst = all_true_pred_thres_change_lst / all_pred_thres_change_lst * 100
        pred_thres_change_percision_lst_print = [round(x, 3) for x in pred_thres_change_percision_lst]
        true_pred_thres_actual_pred_thres_change_precision_lst = all_true_pred_actual_thres_change_lst/ all_pred_thres_change_lst * 100
        true_pred_thres_actual_pred_thres_change_precision_lst_print = [round(x, 3) for x in true_pred_thres_actual_pred_thres_change_precision_lst]
        print('Accuracy List: ', accuracy_lst_print)
        print('Change Accuracy List: ', pred_thres_change_accuracy_lst_print)

        print('Change Direction Pred Precision List: ', pred_thres_change_percision_lst_print)
        print('Pred thres actual change precision: ', true_pred_thres_actual_pred_thres_change_precision_lst_print)
        # print(f'completed in {time.time()-start_time:.2f} seconds')



        if train:
            return average_loss
        else:
            pred_thres_change_accuracy = epoch_true_pred_thres_change / epoch_total_change * 100

            pred_thres_change_precision = sum(pred_thres_change_percision_lst)/len(pred_thres_change_percision_lst)

            pred_thres_actual_change_precision = sum(all_true_pred_actual_thres_change_lst)/sum(all_pred_thres_change_lst) * 100
            # assert tmp == pred_thres_change_actual_pred_thres_change_precision, f'something is wrong: {tmp}, {pred_thres_change_actual_pred_thres_change_precision}'
            pred_thres_actual_change_precision_lst = true_pred_thres_actual_pred_thres_change_precision_lst
            # Define a sample row as a dictionary
            row_dict = {
                'background_up': background_up,
                'up_change_pred_pct': up_change_pred_pct,
                'up_change_pred_precision': up_change_pred_precision,
                'background_dn': background_dn,
                'dn_change_pred_pct': dn_change_pred_pct,
                'dn_change_pred_precision': dn_change_pred_precision,
                'background_none': background_none,
                'none_change_pred_pct': none_change_pred_pct,
                'none_change_pred_precision': none_change_pred_precision,
                'accuracy': accuracy,
                'accuracy_lst': accuracy_lst,
                'pred_thres_change_accuracy': pred_thres_change_accuracy,
                'pred_thres_change_accuracy_lst': pred_thres_change_accuracy_lst,
                'pred_thres_change_precision': pred_thres_change_precision,
                'pred_thres_change_percision_lst': pred_thres_change_percision_lst,
                'pred_thres_actual_change_precision': pred_thres_actual_change_precision,
                'pred_thres_actual_change_precision_lst': pred_thres_actual_change_precision_lst,

                'pred_thres_up_actual_precision': pred_thres_up_actual_precision,
                'pred_thres_dn_actual_precision': pred_thres_dn_actual_precision,
            }
            print(row_dict['accuracy'])
            print(row_dict['pred_thres_change_accuracy'])
            print(row_dict['pred_thres_change_precision'])
            print(row_dict['pred_thres_actual_change_precision'])

            # print(row_dict)
            return true_pred_thres_actual_pred_thres_change_precision_lst, average_loss, row_dict, 
    except KeyboardInterrupt:
        average_loss /= num_epochs
        accuracy_lst = all_true_prediction_lst / all_prediction_lst * 100
        accuracy_lst_print = [round(x, 3) for x in accuracy_lst]
        pred_thres_change_accuracy_lst = all_true_pred_thres_change_lst / actual_thres_change_lst * 100
        pred_thres_change_accuracy_lst_print = [round(x, 3) for x in pred_thres_change_accuracy_lst]

        pred_thres_change_percision_lst = all_true_pred_thres_change_lst / all_pred_thres_change_lst * 100
        pred_thres_change_percision_lst_print = [round(x, 3) for x in pred_thres_change_percision_lst]
        true_pred_thres_actual_pred_thres_change_precision_lst = all_true_pred_actual_thres_change_lst/ all_pred_thres_change_lst * 100
        true_pred_thres_actual_pred_thres_change_precision_lst_print = [round(x, 3) for x in true_pred_thres_actual_pred_thres_change_precision_lst]
        print('Accuracy List: ', accuracy_lst_print)
        print('Change Accuracy List: ', pred_thres_change_accuracy_lst_print)
        print('Change Precision List: ', pred_thres_change_percision_lst_print)
        print('Prediction of Change Precision List: ', true_pred_thres_actual_pred_thres_change_precision_lst_print)
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

def save_params(best_prediction, optimizers, model_state, last_model_pth, best_model_state, best_model_pth, model_training_param_pth, has_improvement, best_k, epoch_num):
    print('saving params...')

    model_lr = get_current_lr(optimizers[0])
    with open(model_training_param_pth, 'w') as f:
        json.dump({'learning_rate': model_lr, 'best_prediction': best_prediction, 'best_k':best_k, 'epoch_num':epoch_num}, f)
    print('saving model to: ', last_model_pth)
    torch.save(model_state, last_model_pth)
    if has_improvement:
        print('saving best model to: ', best_model_pth)
        torch.save(best_model_state, best_model_pth)
    print('done.')
    
def moving_average(data, window_size = 5):
    weights = np.repeat(1.0, window_size) / window_size
    return np.convolve(data, weights, mode='valid')

def ask_for_log(log_pth):
    while True:
        tolog = input(f"Do you want to log the training result? (y/n) ")
        if tolog == 'y':
            print_n_log(log_pth, f'{datetime.now()}')
            purpose = input("What is the purpose/target of this {train}ing? ")
            print_n_log(log_pth, f'purpose: {purpose}')
            return True
        elif tolog == 'n':
            return False

def get_pth(pth_to_main):
    best_model_pth = f'{pth_to_main}/model/model_{config_name}.pt'
    last_model_pth = f'{pth_to_main}/model/last_model_{config_name}.pt'
    model_training_param_pth = f'{pth_to_main}/model/training_param_{config_name}.json'
    return best_model_pth, last_model_pth, model_training_param_pth


def main():
    """
        steps:
        1. log the training purpose
        2. load data and model

    """

    ############################## 1. log the training purpose ##############################
    pth_to_main = '..'
    train_purporse_log_pth = f'{pth_to_main}/log/train_purpose_log.txt'
    should_log_result = ask_for_log(train_purporse_log_pth)

    
    ############################## 2. load data and model ##############################
    # CHANGE CONFIG NAME to save a new model; normally, CONFIG NAME is created autometically in model_structure_param.py
    start_time = time.time()

    print('loading data & model')
    best_model_pth, last_model_pth, model_training_param_pth = get_pth(pth_to_main)

    training_log_pth = f"{pth_to_main}/log/training_log.csv"

    # model, best_model, model_lr, best_prediction, best_k, epoch_nu
    print('loading model')
    start_time = time.time()

    # initalize empty model given model_type (modify in model_structure_param.py)
    if model_type == 'transformer':
        src_pad_idx = -1e20
        trg_pad_idx = -1e20
        src_vocab_size = 10
        trg_vocab_size = 1
        model = Transformer(
            src_vocab_size,
            trg_vocab_size,
            src_pad_idx,
            trg_pad_idx,
            feature_num=feature_num,
            num_layers=num_layers,
            forward_expansion=2, # 4
            heads=4, # 8
            dropout=0,
            device="cuda",
            max_length=hist_window,
        ).to(device)
    elif model_type == 'lstm':
        model = Seq2Seq(input_size, hidden_size, num_layers, output_size, prediction_window, dropout, device, attention = True).to(device)
        # model = Seq2SeqDirectionClassification(input_size, hidden_size, num_layers, output_size, prediction_window, dropout, device, attention = True).to(device)
    best_model = copy.deepcopy(model)

    if os.path.exists(last_model_pth):
        print('Loading existing model')
        model.load_state_dict(torch.load(last_model_pth))
        best_model.load_state_dict(torch.load(best_model_pth))
        with open(model_training_param_pth, 'r') as f:
            saved_data = json.load(f)
            model_lr = saved_data['learning_rate']
            best_prediction = saved_data['best_prediction']
            best_k = saved_data['best_k']
            epoch_num = saved_data['epoch_num']
            print('best_k: ', best_k)
            start_best_prediction = best_prediction
    # elif os.path.exists(best_model_pth):
    else:
        print('No existing model')
        model_lr = learning_rate
        best_prediction = 0.0
        best_k = 0.0
        epoch_num = 0
        start_best_prediction = 0.0
    best_model_state = best_model.state_dict()
   
    print(model)
    print(f'model loading completed in {time.time()-start_time:.2f} seconds')


    # optimizer = SGD(model.parameters(), lr=learning_rate)
    optimizer = AdamW(model.parameters(),weight_decay=1e-5, lr=model_lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience, threshold=scheduler_threshold)
    optimizers = [optimizer]
    schedulers = [scheduler]
    

    
    
    
    # TODO: automatically change training set; 
    # Partially done; 
    # but it only change target after an epoch; which might be problematic...
    # since each dataset is so large.
    try:
        plt.ion
        # Train the model
        start_time = time.time()
        print('Training model')
        test_every_x_epoch = 1
        plot_hist_dict = {}
        fig, (ax_0, ax_1, ax_2) = plt.subplots(3,1, figsize=(10,10))
        fig.subplots_adjust(top=0.95, bottom=0.05)
        manager = plt.get_current_fig_manager()
        manager.full_screen_toggle()
        # test_pred_thres_change_precision_hist = np.zeros((prediction_window,1))
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
        
        ma_weight_decay = weight_decay
        for epoch in range(num_epochs):
            symbol = random.choice(training_symbols)
            print("Symbol for the epoch: ", symbol)
            data_load_start_time = time.time()
            data_pth = f'{pth_to_main}/data/csv/training/bar_set_20200101_20200701_{symbol}_1Min_{feature_num}feature0.csv'
            train_loader, test_loader = load_n_split_data(data_pth, hist_window, prediction_window, batch_size, train_ratio)
            print(f'data loading completed in {time.time()-data_load_start_time:.2f} seconds')
            print(f'Epoch {epoch+1}/{num_epochs}')
            if test_every_x_epoch != 0 and epoch % test_every_x_epoch == 0:
                with torch.no_grad():
                    true_pred_thres_actual_pred_thres_change_precision_lst, average_loss, row_dict = work(model, test_loader, optimizers, num_epochs = 1, train = False)
                    for key, value in row_dict.items():
                        if key not in plot_hist_dict:
                            plot_hist_dict[key] = []
                        plot_hist_dict[key].append(value)
                    plot_history(plot_hist_dict, ax_0, ax_1, ax_2)
                    
                    row_dict['model_pth'] = last_model_pth
                    row_dict['time'] = datetime.now()
                    row_dict['best_k'] = best_k
                    row_dict['epoch_num'] = epoch_num
                    if not os.path.exists(training_log_pth):
                        # If not, create the file with the header
                        with open(training_log_pth, 'w', newline='') as csvfile:
                            csv_writer = csv.DictWriter(csvfile, fieldnames=header)
                            csv_writer.writeheader()

                    # Append the new row to the CSV file
                    with open(training_log_pth, 'a', newline='') as csvfile:
                        csv_writer = csv.DictWriter(csvfile, fieldnames=header)
                        csv_writer.writerow(row_dict)


                    epoch_best_weights = weights_lst[0]
                    highest_pred_thres_change_precision = 0.0
                    improved = False
                    epoch_best_k = -100
                    for k, weights in enumerate(weights_lst):
                        # print(weights)
                        pred_thres_change_precision = true_pred_thres_actual_pred_thres_change_precision_lst.reshape(1,prediction_window)*weights
                        pred_thres_change_precision = pred_thres_change_precision.sum()/np.sum(weights)
                        if pred_thres_change_precision > highest_pred_thres_change_precision:
                            highest_pred_thres_change_precision = pred_thres_change_precision
                            epoch_best_k = k
                            epoch_best_weights = weights
                        if pred_thres_change_precision > best_prediction: 
                            best_k = k
                            has_improvement = True
                            improved = True
                            print(epoch_best_weights)
                            print(f'\nNEW BEST prediction: {pred_thres_change_precision:.4f}% at k: {best_k}\n')
                            best_prediction = pred_thres_change_precision
                            best_model_state = model.state_dict()
                    tmp_weight_decay = pow(0.8, epoch_best_k)
                    ma_weight_decay = tmp_weight_decay * 0.1 + ma_weight_decay * 0.9
                    global train_weights
                    train_weights = torch.pow(torch.tensor(ma_weight_decay), torch.arange(prediction_window).float()).to(device)
                    if not improved:
                        print(epoch_best_weights)
                        print(f'\ncurrent best change prediction precision: {highest_pred_thres_change_precision:.4f}% at k: {epoch_best_k}\n')
            # endif

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

        print(test_pred_thres_change_precision_hist.shape)
        # predictions = np.mean(predictions, axis = )
        # plt.ion()
        # for i in range(prediction_window):
        #     predictions = test_pred_thres_change_precision_hist[i,:]
        #     plt.plot(predictions, label=f'{i+1} min prediction', linestyle='solid')
        # plt.plot(test_pred_thres_change_precision_hist.mean(axis=0), label=f'average prediction', linestyle='dashed')
        # weights = np.linspace(1, 0.1, num=prediction_window)
        # weights = weights.reshape(prediction_window,1)
        # plt.plot((test_pred_thres_change_precision_hist*weights).sum(axis=0)/np.sum(weights), label=f'weighted prediction', linestyle='dotted')
        # plt.legend()

        print('Training Complete')
        # plt.show()
        # plt.clf()
        # plt.ioff()

        model_lr = get_current_lr(optimizer)
        lrs = [model_lr]

        # Test the model
        # start_time = time.time()
        # print('Testing model')
        # with torch.no_grad():
        #     test_pred_thres_change_percision_lst, average_loss = work(model, test_loader, optimizers, num_epochs = 1, mode = 1)
        #     test_pred_thres_change_precision_hist[:,0] = test_pred_thres_change_percision_lst
        #     plt.clf()
        #     for i in range(prediction_window):
        #         predictions = test_pred_thres_change_precision_hist[i,:]
        #         plt.plot(predictions, 0, label=f'{i+1} min accuracy', marker = 'o')
        #     plt.plot(test_pred_thres_change_precision_hist.mean(axis=0), 0, label=f'average accuracy', marker = 'o')
        #     plt.plot((test_pred_thres_change_precision_hist*weights).sum(axis=0)/np.sum(weights), label=f'weighted accuracy', linestyle='dotted')
        #     plt.legend()
        #     plt.ylim(0, 1)
        #     plt.show()
        #     plt.pause(1)
        # print(f'testing completed in {time.time()-start_time:.2f} seconds')
        last_model_state = model.state_dict()
        save_params(best_prediction, optimizers, last_model_state, last_model_pth, best_model_state, best_model_pth, model_training_param_pth, has_improvement, best_k, epoch_num) 
        print('Normal exit. Model saved.')
        torch.cuda.empty_cache()
        gc.collect()
    except KeyboardInterrupt or Exception or TypeError:
        # save the model if the training was interrupted by keyboard input
        last_model_state = model.state_dict()
        save_params(best_prediction, optimizers, last_model_state, last_model_pth, best_model_state, best_model_pth, model_training_param_pth, has_improvement, best_k, epoch_num)
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == '__main__':
    main()
    # cProfile.run('main()') # this shows execution time of each function. Might be useful for debugging & accelerating in detail.