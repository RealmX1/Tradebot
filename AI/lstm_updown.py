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
import sys
sys.path.append('..')
from common import *

np.set_printoptions(precision=4, suppress=True) 


# import custom files
from S2S import *
# from sim import *
from data_utils import * 
from model_structure_param import * # Define hyperparameters

train_ratio     = 0.9
num_epochs      = 100

loss_fn = nn.MSELoss(reduction = 'none')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

training_log_pth = 'training_log/{{}}'


torch.autograd.set_detect_anomaly(True)


def get_direction_diff(y_batch,y_pred):
    

    # start_time = time.time()
    # print_n_log('start get_direction_diff')
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

    # print_n_log('True: ', true_direction.shape)
    # print_n_log('Pred: ', pred_direction)

    instance_num =  true_direction.shape[0]
    prediction_min = true_direction.shape[1]

    all_cells = instance_num * prediction_min
    same_cells = np.count_nonzero(true_direction == pred_direction)

    all_cells_lst = np.full((prediction_min,), instance_num)
    same_cells_lst = np.count_nonzero(true_direction == pred_direction, axis = 0)

    all_change_lst = np.count_nonzero(true_direction != 0, axis = 0)
    true_change_pred_lst = np.count_nonzero((true_direction == pred_direction) & (true_direction != 0), axis = 0)
    all_change_pred_lst = np.count_nonzero(pred_direction != 0, axis = 0)

    all_change = np.sum(all_change_lst)
    true_change_pred = np.sum(true_change_pred_lst)
    all_change_pred = np.sum(all_change_pred_lst)
    # print_n_log('all_cells: ',all_cells)
    # print_n_log('same_cells.shape: ',same_cells.shape)
    # print_n_log(type(true_direction))
    # print_n_log(true_direction.typedf())
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

    # print_n_log('all_cells: ',all_cells)
    # print_n_log('all_cells_lst: ',all_cells_lst)
    # print_n_log('same_cells: ',same_cells)
    # print_n_log('same_cells_lst: ',same_cells_lst)
    # print_n_log('all_true: ',tp+tn)
    # print_n_log('all_false: ',fp+fn)
    # print_n_log('all_num: ', tp+tn+fp+fn)

    # print_n_log('get_direction_diff time: ', time.time()-start_time)
    all_pred_actual_change_lst = np.sum(pred_direction != 0, axis = 0)
    true_pred_actual_change_lst = np.sum((actual_direction == pred_direction) & (pred_direction != 0), axis = 0)
    all_pred_actual_change = np.sum(all_pred_actual_change_lst)
    true_pred_actual_change = np.sum(true_pred_actual_change_lst)
    return all_cells, same_cells, \
            all_cells_lst, same_cells_lst, \
            all_change, true_change_pred, all_change_pred, \
            all_change_lst, true_change_pred_lst, all_change_pred_lst,\
            t_up, f_up, t_dn, f_dn, t_below_thres, f_below_thres, \
            all_up, all_down, all_below_thres, \
            all_pred_actual_change, true_pred_actual_change, \
            all_pred_actual_change_lst, true_pred_actual_change_lst


def calculate_policy_return(x_batch,y_batch,y_pred):
    pass

def count_tensor_num():
    num = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                num += 1
                # print_n_log(type(obj), obj.size())
        except:
            pass
    print_n_log('num of tensors: ', num)

def print_strs(strs):
    general_stat_str=   f'Epoch {epoch+1:3}/{num_epochs:3}, ' + \
                        f'Loss: {                   epoch_loss:10.7f}, ' + \
                        f'Time/epoch: {             (time.time()-start_time)/(epoch+1):.2f} seconds, ' + \
                        f'\u2713 Direction: {       accuracy:.2f}%, ' + \
                        f'\u2713 change Direction: {change_accuracy:.2f}%, ' + \
                        f'Encocder LR: {            get_current_lr(optimizers[0]):9.8f},' # Decoder LR: {get_current_lr(optimizers[1]):9.8f}, '
            
        change_up_str = f'Background \u2191: {  epoch_up        /epoch_predictions*100:7.4f}%, ' + \
                        f'\u2191 Pred pct: {    epoch_up_pred   /epoch_predictions*100:7.4f}%, ' + \
                        f'\u2191 Precision: {   epoch_t_up      /epoch_up_pred*100:7.4f}%, '

        change_down_str=f'Background \u2193: {  epoch_down      /epoch_predictions*100:7.4f}%, ' + \
                        f'\u2193 Pred pct: {    epoch_down_pred /epoch_predictions*100:7.4f}%, ' + \
                        f'\u2193 Precision: {   epoch_t_dn      /epoch_down_pred*100:7.4f}%, '
            
        change_none_str=f'Background \u2192: {  epoch_below_thres       /epoch_predictions*100:7.4f}%, ' + \
                        f'\u2192 Pred pct: {    epoch_below_thres_pred  /epoch_predictions*100:7.4f}%, ' + \
                        f'\u2192 Precision: {   epoch_t_below_thres     /epoch_below_thres_pred*100:7.4f}%'
        print_n_log(general_stat_str)
        print_n_log(change_up_str)
        print_n_log(change_down_str)
        print_n_log(change_none_str)

weights = torch.pow(torch.tensor(weight_decay), torch.arange(prediction_window).float()).to(device)
print_n_log("weights: ",weights.shape)

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


        all_predictions             = np.zeros(prediction_window) # one elemnt for each minute of prediction window
        all_true_predictions        = np.zeros(prediction_window)
        all_changes               = np.zeros(prediction_window)
        all_true_change_predictions = np.zeros(prediction_window)
        all_change_predictions      = np.zeros(prediction_window)

        all_all_pred_actual_change_lst = np.zeros(prediction_window)
        all_true_pred_actual_change_lst = np.zeros(prediction_window)

        average_loss = 0

        inverse_mask = torch.linspace(1, 11, 10)
        # print ('inverse_mask.shape: ', inverse_mask.shape)
        global weights
        # print_n_log(weights)
        # ([1.0000, 0.8000, 0.6400, 0.5120, 0.4096, 0.3277, 0.2621, 0.2097, 0.1678,0.1342])
        # weights = torch.linspace(1, 0.1, steps=prediction_window)
        # ma_loss = None
        for epoch in range(num_epochs):
            epoch_loss = 0
            epoch_predictions = 0
            epoch_true_predictions = 0
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
                # print_n_log(f"There are currently {tensor_count} Torch tensors in memory.")
                # block_idx = [0,1,4,5,18,19] # it seems that open isn't useful at all.... since it is basically the last close price.
                # x_batch[:,:,block_idx] = 0
                
                # print_n_log('x_batch[0,:,:]: ', x_batch[0,:,:])
                # x_batch   [N, hist_window, feature_num]
                # y_batch & output  [N, prediction_window]
                x_batch = x_batch.float().to(device) # probably need to do numpy to pt tensor in the dataset; need testing on efficiency #!!! CAN'T BE DONE. before dataloarder they are numpy array, not torch tensor
                # print_n_log('one input: ', x_batch[0:,:,:])
                y_batch = y_batch.float().to(device)
                
                y_pred = model(x_batch, y_batch, teacher_forcing_ratio) # [N, prediction_window]
                # print_n_log('y_pred.shape: ', y_pred.shape)
                # print_n_log('y_batch.shape: ', y_pred.shape)
                # print_n_log('y_batch: ', y_batch[0,:])
                loss = loss_fn(y_pred, y_batch)
                # print_n_log('loss shape: ', loss.shape)

                loss_val = loss.clone().detach().cpu().mean().item()

                weighted_loss = loss * weights
                # print_n_log(weighted_loss.shape)
                # print_n_log(f"loss: {loss.shape}, weighted loss: {weighted_loss.shape}")
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
                # print_n_log('memory_used: ', memory_used/1024/1024/8)

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
                ac, tcp, acp, \
                acl, tcpl, acpl, \
                tp, fp, tn, fn, t_below_thres, f_below_thres, \
                all_up, all_down, all_below_thres, \
                apac, tpac, \
                apacl, tpacl = get_direction_diff(y_batch_safe, y_pred_safe)
                
                epoch_predictions += all_cells
                epoch_true_predictions += same_cells
                epoch_t_up += tp
                epoch_f_up += fp
                epoch_t_dn += tn
                epoch_f_dn += fn
                epoch_t_below_thres += t_below_thres
                epoch_f_below_thres += f_below_thres

                epoch_up += all_up
                epoch_down += all_down
                epoch_below_thres += all_below_thres


                all_predictions += all_cells_lst
                all_true_predictions += same_cells_lst
                all_changes += acl
                all_true_change_predictions += tcpl
                all_change_predictions += acpl

                all_all_pred_actual_change_lst += apacl
                all_true_pred_actual_change_lst += tpacl

                
                epoch_loss += loss_val
                
            epoch_loss /= (i+1)
            average_loss += epoch_loss
            accuracy = epoch_true_predictions / epoch_predictions * 100
            change_accuracy = tpac / apac * 100
            assert epoch_t_up + epoch_f_up + epoch_t_dn + epoch_f_dn + epoch_t_below_thres + epoch_f_below_thres == epoch_predictions
            assert epoch_up + epoch_down + epoch_below_thres == epoch_predictions
            # assert epoch_t_up + epoch_f_dn == epoch_up No longer applicable after adding below_thres
            # assert epoch_f_up + epoch_t_dn == epoch_down

            epoch_up_pred = epoch_t_up + epoch_f_up
            epoch_down_pred = epoch_f_dn + epoch_t_dn
            epoch_below_thres_pred = epoch_t_below_thres + epoch_f_below_thres
            
            
                # f'Weighted Loss: {final_loss.item():10.7f}, MA Loss: {ma_loss.mean().item():10.7f}') 
                
            del x_batch, y_batch, y_pred, loss, weighted_loss, final_loss
        
        average_loss /= num_epochs
        accuracy_lst = all_true_predictions / all_predictions * 100
        accuracy_lst_print = [round(x, 3) for x in accuracy_lst]

        change_accuracy_lst = all_true_change_predictions / all_changes * 100
        change_accuracy_lst_print = [round(x, 3) for x in change_accuracy_lst]
        change_precision_lst = all_true_change_predictions / all_change_predictions * 100
        change_precision_lst_print = [round(x, 3) for x in change_precision_lst]
        prediction_of_change_precision_lst = all_true_pred_actual_change_lst/ all_all_pred_actual_change_lst * 100
        prediction_of_change_precision_lst_print = [round(x, 3) for x in prediction_of_change_precision_lst]
        print_n_log('Accuracy List: ', accuracy_lst_print)
        print_n_log('Change Accuracy List: ', change_accuracy_lst_print)
        print_n_log('Change Precision List: ', change_precision_lst_print)
        print_n_log('Prediction of Change Precision List: ', prediction_of_change_precision_lst_print)
        # print_n_log(f'completed in {time.time()-start_time:.2f} seconds')

        if train:
            return average_loss
        else:
            return prediction_of_change_precision_lst, average_loss
    except KeyboardInterrupt:
        average_loss /= num_epochs
        accuracy_lst = all_true_predictions / all_predictions * 100
        accuracy_lst_print = [round(x, 3) for x in accuracy_lst]

        change_accuracy_lst = all_true_change_predictions / all_changes * 100
        change_accuracy_lst_print = [round(x, 3) for x in change_accuracy_lst]
        change_precision_lst = all_true_change_predictions / all_change_predictions * 100
        change_precision_lst_print = [round(x, 3) for x in change_precision_lst]
        prediction_of_change_precision_lst = all_true_pred_actual_change_lst/ all_all_pred_actual_change_lst * 100
        prediction_of_change_precision_lst_print = [round(x, 3) for x in prediction_of_change_precision_lst]
        print_n_log('Accuracy List: ', accuracy_lst_print)
        print_n_log('Change Accuracy List: ', change_accuracy_lst_print)
        print_n_log('Change Precision List: ', change_precision_lst_print)
        print_n_log('Prediction of Change Precision List: ', prediction_of_change_precision_lst_print)

        print_n_log('CSV FRIENDLY:')
        print_n_log('')

        raise KeyboardInterrupt# re-raise the exception



def plot(predictions, targets, test_size):
    # Plot the results
    print_n_log('total entry: ',predictions.shape[0])
    x = np.arange(len(predictions))
    print_n_log('predictions.shape: ',predictions.shape)
    plt.plot(targets, label='Actual')
    plt.plot(predictions, label='Predicted',linestyle='dotted')
    plt.legend()
    plt.axvline(x=test_size, color='r')
    plt.show()

def get_current_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def save_params(best_prediction, optimizers, model_state, last_model_pth, best_model_state, best_model_pth, model_training_param_path, has_improvement, best_weight_decay):
    print_n_log('saving params...')

    encoder_lr = get_current_lr(optimizers[0])
    decoder_lr = get_current_lr(optimizers[1])
    with open(model_training_param_path, 'w') as f:
        json.dump({'encoder_learning_rate': encoder_lr, 'decoder_learning_rate': decoder_lr, 'best_prediction': best_prediction, 'best_weight_decay':best_weight_decay}, f)
    print_n_log('saving model to: ', last_model_pth)
    torch.save(model_state, last_model_pth)
    if has_improvement:
        print_n_log('saving best model to: ', best_model_pth)
        torch.save(best_model_state, best_model_pth)
    print_n_log('done.')
    
def moving_average(data, window_size = 5):
    weights = np.repeat(1.0, window_size) / window_size
    return np.convolve(data, weights, mode='valid')

def main():
    # CHANGE CONFIG NAME to save a new model
    start_time = time.time()
    print_n_log('loading data & model')
    # data_pth = 'data/cdl_test_2.csv'
    data_pth = training_data_path
    best_model_pth = f'../model/model_{config_name}.pt'
    best_weight_decay = 0.00

    last_model_pth = f'../model/last_model_{config_name}.pt'
    model_training_param_path = f'../model/training_param_{config_name}.json'
    print_n_log('loaded in ', time.time()-start_time, ' seconds')
    
    train_loader, test_loader = load_n_split_data(data_pth, hist_window, prediction_window, batch_size, train_ratio)
    


    print_n_log('loading model')
    start_time = time.time()
    model = Seq2Seq(input_size, hidden_size, num_layers, output_size, prediction_window, dropout, device, attention = True).to(device)
    best_model = copy.deepcopy(model)
    if os.path.exists(last_model_pth):
        print_n_log('Loading existing model')
        model.load_state_dict(torch.load(last_model_pth))
        best_model.load_state_dict(torch.load(best_model_pth))
        with open(model_training_param_path, 'r') as f:
            saved_data = json.load(f)
            encoder_lr = saved_data['encoder_learning_rate']
            decoder_lr = saved_data['decoder_learning_rate']
            best_prediction = saved_data['best_prediction']
            start_best_prediction = best_prediction
    else:
        print_n_log('No existing model')
        encoder_lr = learning_rate
        decoder_lr = learning_rate
        best_prediction = 0.0
        start_best_prediction = best_prediction
    best_model_state = best_model.state_dict()
   
    print_n_log(model)
    print_n_log(f'model loading completed in {time.time()-start_time:.2f} seconds')


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
        print_n_log('Training model')
        test_every_x_epoch = 1
        test_change_precision_hist = np.zeros((prediction_window,1))
        eval_loss_hist = []
        train_loss_hist = []

        has_improvement = False

        arr = np.ones(prediction_window)
        for i in range(1, prediction_window):
            arr[i] = arr[i-1] * weight_decay
        weights = arr.reshape(1,prediction_window)
        
        for epoch in range(num_epochs):
            print_n_log(f'Epoch {epoch+1}/{num_epochs}')
            if test_every_x_epoch != 0 and epoch % test_every_x_epoch == 0:
                with torch.no_grad():
                    prediction_of_change_precision_lst, average_loss = work(model, test_loader, optimizers, num_epochs = 1, train = False)
                    
                    # print_n_log(test_change_precision_lst.shape)
                    change_precision = prediction_of_change_precision_lst.reshape(1,prediction_window)*weights
                    change_precision = change_precision.sum()/np.sum(weights)
                    # maybe i should try iterate through different weights at this stage to find the best weight for the stage. (and thus potentialy find the best possible model -- remmber to record this best weight in the best model's name.)
                    # maybe i should try use the 
                    if epoch == 0:
                        test_change_precision_hist[:,0] = prediction_of_change_precision_lst
                    else:
                        test_change_precision_hist = np.concatenate((test_change_precision_hist, prediction_of_change_precision_lst.reshape(prediction_window,1)), axis=1)
                    eval_loss_hist.append(average_loss)

                    for k in range(-10,11):
                        arr = np.ones(prediction_window)
                        w_d = 0.0
                        for i in range(1, prediction_window):
                            w_d = pow(0.8,k)
                            arr[i] = arr[i-1] * w_d
                        weights = arr.reshape(1,prediction_window)
                        change_precision = prediction_of_change_precision_lst.reshape(1,prediction_window)*weights
                        change_precision = change_precision.sum()/np.sum(weights)
                        if change_precision > best_prediction: 
                            best_weight_decay = w_d
                            has_improvement = True
                            print_n_log(f'\nNEW BEST prediction: {change_precision:.4f}% at weight decay: {w_d}\n')
                            best_prediction = change_precision
                            best_model_state = model.state_dict()
                        else:
                            print_n_log(f'current change prediction precision: {change_precision:.4f}%')
                
                    plt.clf()
                    for i in range(prediction_window):
                        accuracies = test_change_precision_hist[i,:]
                        plt.plot(accuracies, label=f'{i+1} min accuracy', linestyle='solid')
                    plt.plot(test_change_precision_hist.mean(axis=0), label=f'average accuracy', linestyle='dashed')
                    weighted_accuracy_lst = np.matmul(weights,test_change_precision_hist)/np.sum(weights)
                    # print_n_log('weighted_accuracy_lst', weighted_accuracy_lst.shape) # shape (1, epoch)
                    plt.plot(weighted_accuracy_lst[0], label=f'weighted accuracy', linestyle='dotted')
                # plt.clf()
                # plt.plot(moving_average(eval_loss_hist, 3), label=f'loss', linestyle='solid')

                # actually train the model
            average_loss = work(model, train_loader, optimizers, test_every_x_epoch, train = True, schedulers = schedulers)
                # train_loss_hist.append(average_loss)
                # plt.plot(train_loss_hist, label=f'train loss', linestyle='dotted')
            # if epoch == 0:
            plt.legend() 
            plt.pause(0.5)
        print_n_log(f'training completed in {time.time()-start_time:.2f} seconds')
        
        print_n_log('\n\n')
        if best_prediction > start_best_prediction:
            print_n_log(f'improved from {start_best_prediction:.2f}% to {best_prediction:.2f}%')
        else:
            print_n_log(f'NO IMPROVEMENET from {start_best_prediction:.2f}%')
        print_n_log('\n\n')

        print_n_log(test_change_precision_hist.shape)
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

        print_n_log('Training Complete')
        plt.show()
        plt.clf()
        # plt.ioff()

        encoder_lr = get_current_lr(encoder_optimizer)
        decoder_lr = get_current_lr(decoder_optimizer)
        lrs = [encoder_lr, decoder_lr]

        # Test the model
        # start_time = time.time()
        # print_n_log('Testing model')
        # with torch.no_grad():
        #     test_change_precision_lst, average_loss = work(model, test_loader, optimizers, num_epochs = 1, mode = 1)
        #     test_change_precision_hist[:,0] = test_change_precision_lst
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
        # print_n_log(f'testing completed in {time.time()-start_time:.2f} seconds')

        save_params(best_prediction, optimizers, model.state_dict(), last_model_pth, best_model_state, best_model_pth, model_training_param_path, has_improvement, best_weight_decay) 
        print_n_log('Normal exit. Model saved.')
        torch.cuda.empty_cache()
        gc.collect()
    except KeyboardInterrupt or Exception or TypeError:
        # save the model if the training was interrupted by keyboard input
        save_params(best_prediction, optimizers, model.state_dict(), last_model_pth, best_model_state, best_model_pth, model_training_param_path, has_improvement, best_weight_decay)
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == '__main__':
    main()
    # cProfile.run('main()') # this shows execution time of each function. Might be useful for debugging & accelerating in detail.


def on_exit():
    # close all open figures
    plt.close('all')
    
    # destroy the Tkinter application
    root.destroy()