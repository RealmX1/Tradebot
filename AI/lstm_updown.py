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


torch.autograd.set_detect_anomaly(True)


def get_direction_diff(y_batch,y_pred):
    

    # start_time = time.time()
    # print('start get_direction_diff')
    # true_direction = y_batch-x_batch[:,-1,close_idx:close_idx+1]
    y_batch_np_raw = y_batch.cpu().numpy()
    y_batch_below_threshold = np.zeros_like(y_batch_np_raw, dtype=bool)
    y_batch_below_threshold[np.abs(y_batch_np_raw) < policy_threshold] = True
    true_direction = np.clip(y_batch_np_raw, 0, np.inf) # this turns negative to 0, positive to 1
    true_direction[true_direction != 0] = 1
    true_direction[true_direction == 0] = -1
    true_direction[y_batch_below_threshold] = 0
    # true_direction = y_batch.cpu().numpy()

    y_pred_np_raw = y_pred.detach().cpu().numpy()
    y_pred_below_threshold = np.zeros_like(y_pred_np_raw, dtype=bool)
    y_pred_below_threshold[np.abs(y_pred_np_raw) < policy_threshold] = True
    pred_direction = np.clip(y_pred_np_raw, 0, np.inf) # turn all 
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

    true_change_lst = np.count_nonzero(true_direction != 0, axis = 0)
    true_change_true_pred_lst = np.count_nonzero((true_direction == pred_direction) & (true_direction != 0), axis = 0)
    true_change_all_pred_lst = np.count_nonzero(pred_direction != 0, axis = 0)

    true_change = np.sum(true_change_lst)
    true_change_true_pred = np.sum(true_change_true_pred_lst)
    true_change_all_pred = np.sum(true_change_all_pred_lst)
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
    return all_cells, same_cells, \
            true_change, true_change_true_pred, true_change_all_pred, \
            all_cells_lst, same_cells_lst, \
            true_change_lst, true_change_true_pred_lst, true_change_all_pred_lst,\
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

def work(model, data_loader, optimizers, num_epochs = num_epochs, mode = 0, schedulers = None): # mode 0: train, mode 1: test, mode 2: PLOT
    if mode == 0:
        teacher_forcing_ratio = 0.1
        model.train()
    else:
        teacher_forcing_ratio = 0
        model.eval()
    start_time = time.time()
    same_cells = 0

    # count_tensor_num()


    all_predictions             = np.zeros(prediction_window) # one elemnt for each minute of prediction window
    all_true_predictions        = np.zeros(prediction_window)
    all_changes                 = np.zeros(prediction_window)
    all_change_true_predictions = np.zeros(prediction_window)
    all_change_all_predictions  = np.zeros(prediction_window)

    average_loss = 0

    inverse_mask = torch.linspace(1, 11, 10)
    # print ('inverse_mask.shape: ', inverse_mask.shape)
    weights = torch.pow(torch.tensor(weight_decay), torch.arange(prediction_window).float()).to(device)
    # print(weights)
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
        for i, (x_batch, y_batch, x_raw_close) in enumerate(data_loader):
            # block_idx = [0,1]
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

            loss_val = loss.mean().item()

            weighted_loss = loss * weights
            final_loss = weighted_loss.mean()
            
            if mode == 0:
                for optimizer in optimizers:
                    optimizer.zero_grad() # removing zero_grad doesn't improve training speed (unlike some claimed); need more testing
                final_loss.backward()
                for optimizer in optimizers:
                    optimizer.step()
                if schedulers is not None:
                    for scheduler in schedulers:
                        scheduler.step(final_loss)
                        
                        # pass

            # tmp = weighted_loss.detach()
            # if ma_loss is None:
            #     ma_loss = tmp.sum(axis = 0)
            # else:
            #     ma_loss *= 0.8
            #     ma_loss += 0.2*tmp.sum(axis = 0)

            all_cells, same_cells, \
            tc, tctp, tcap, \
            all_cells_lst, same_cells_lst, \
            tcl, tctpl, tcapl, \
            tp, fp, tn, fn, t_below_thres, f_below_thres, up, down, below_thres = get_direction_diff(y_batch, y_pred)
            
            epoch_predictions += all_cells
            epoch_true_predictions += same_cells
            epoch_t_up += tp
            epoch_f_up += fp
            epoch_t_dn += tn
            epoch_f_dn += fn
            epoch_t_below_thres += t_below_thres
            epoch_f_below_thres += f_below_thres

            epoch_up += up
            epoch_down += down
            epoch_below_thres += below_thres
            all_predictions += all_cells_lst
            all_true_predictions += same_cells_lst
            all_changes += tcl
            all_change_true_predictions += tctpl
            all_change_all_predictions += tcapl


            
            epoch_loss += loss_val
            
        epoch_loss /= (i+1)
        average_loss += epoch_loss
        accuracy = epoch_true_predictions / epoch_predictions * 100
        assert epoch_t_up + epoch_f_up + epoch_t_dn + epoch_f_dn + epoch_t_below_thres + epoch_f_below_thres == epoch_predictions
        assert epoch_up + epoch_down + epoch_below_thres == epoch_predictions
        # assert epoch_t_up + epoch_f_dn == epoch_up No longer applicable after adding below_thres
        # assert epoch_f_up + epoch_t_dn == epoch_down

        epoch_up_pred = epoch_t_up + epoch_f_up
        epoch_down_pred = epoch_f_dn + epoch_t_dn
        epoch_below_thres_pred = epoch_t_below_thres + epoch_f_below_thres
            
        print(f'Epoch {epoch+1:3}/{num_epochs:3}, ' +
              f'Loss: {epoch_loss:10.7f}, ' +
              f'Time/epoch: {(time.time()-start_time)/(epoch+1):.2f} seconds, ' +
              f'\u2713 Direction: {accuracy:.2f}%, ' +
              f'Encocder LR: {get_current_lr(optimizers[0]):9.8f},' + # Decoder LR: {get_current_lr(optimizers[1]):9.8f}, ' +
              f'\nBackground \u2191: {  epoch_up        /epoch_predictions*100:7.4f}%, ' +
              f'\u2191 Pred pct: {      epoch_up_pred   /epoch_predictions*100:7.4f}%, ' +
              f'\u2191 Precision: {     epoch_t_up      /epoch_up_pred*100:7.4f}%, ' +

              f'\nBackground \u2193: {  epoch_down      /epoch_predictions*100:7.4f}%, ' +
              f'\u2193 Pred pct: {      epoch_down_pred /epoch_predictions*100:7.4f}%, ' +
              f'\u2193 Precision: {     epoch_t_dn      /epoch_down_pred*100:7.4f}%, ' +
              
              f'\nBackground \u2192: {  epoch_below_thres       /epoch_predictions*100:7.4f}%, ' +
              f'\u2192 Pred pct: {      epoch_below_thres_pred  /epoch_predictions*100:7.4f}%, ' +
              f'\u2192 Precision: {     epoch_t_below_thres     /epoch_below_thres_pred*100:7.4f}%')
              # f'Weighted Loss: {final_loss.item():10.7f}, MA Loss: {ma_loss.mean().item():10.7f}') 
            
    
    average_loss /= num_epochs
    accuracy_lst = all_true_predictions / all_predictions * 100
    accuracy_lst_print = [round(x, 3) for x in accuracy_lst]

    change_accuracy_lst = all_change_true_predictions / all_changes * 100
    change_accuracy_lst_print = [round(x, 3) for x in change_accuracy_lst]
    change_precision_lst = all_change_true_predictions / all_change_all_predictions * 100
    change_precision_lst_print = [round(x, 3) for x in change_precision_lst]
    print('Accuracy List: ', accuracy_lst_print)
    print('Change Accuracy List: ', change_accuracy_lst_print)
    print('Change Precision List: ', change_precision_lst_print)
    # print(f'completed in {time.time()-start_time:.2f} seconds')

    if mode == 1:
        return change_precision_lst, average_loss
    else:
        return average_loss

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

def save_params(best_prediction, optimizers, model_state, best_model_state, model_path, last_model_path, model_training_param_path, has_improvement = True):
    print('saving params...')

    encoder_lr = get_current_lr(optimizers[0])
    decoder_lr = get_current_lr(optimizers[1])
    with open(model_training_param_path, 'w') as f:
        json.dump({'encoder_learning_rate': encoder_lr, 'decoder_learning_rate': decoder_lr, 'best_prediction': best_prediction}, f)
    print('saving model...')
    if has_improvement:
        torch.save(best_model_state, model_path)
    torch.save(model_state, last_model_path)
    print('done.')
    
def moving_average(data, window_size = 5):
    weights = np.repeat(1.0, window_size) / window_size
    return np.convolve(data, weights, mode='valid')

def main():
    # CHANGE CONFIG NAME to save a new model
    start_time = time.time()
    print('loading data & model')
    # data_path = 'data/cdl_test_2.csv'
    data_path = '../data/csv/bar_set_huge_20200101_20230417_AAPL_23feature.csv'
    model_path = f'../model/model_{config_name}.pt'
    last_model_path = f'../model/last_model_{config_name}.pt'
    model_training_param_path = f'../model/training_param_{config_name}.json'
    print('loaded in ', time.time()-start_time, ' seconds')
    
    train_loader, test_loader = load_n_split_data(data_path, hist_window, prediction_window, batch_size, train_ratio)
    


    print('loading model')
    start_time = time.time()
    model = Seq2Seq(input_size, hidden_size, num_layers, output_size, prediction_window, dropout, device, attention = True).to(device)
    best_model = copy.deepcopy(model)
    if os.path.exists(last_model_path):
        print('Loading existing model')
        model.load_state_dict(torch.load(last_model_path))
        best_model.load_state_dict(torch.load(model_path))
        with open(model_training_param_path, 'r') as f:
            saved_data = json.load(f)
            encoder_lr = saved_data['encoder_learning_rate']
            decoder_lr = saved_data['decoder_learning_rate']
            best_prediction = saved_data['best_prediction']
            start_best_prediction = best_prediction
    else:
        print('No existing model')
        encoder_lr = learning_rate
        decoder_lr = learning_rate
        best_prediction = 0.0
        start_best_prediction = best_prediction
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
        test_change_precision_hist = np.zeros((prediction_window,1))
        eval_loss_hist = []
        train_loss_hist = []

        has_improvement = False

        arr = np.ones(prediction_window)
        for i in range(1, prediction_window):
            arr[i] = arr[i-1] * weight_decay
        weights = arr.reshape(prediction_window,1)
        
        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}/{num_epochs}')
            if test_every_x_epoch and epoch % test_every_x_epoch == 0:
                test_change_precision_lst, average_loss = work(model, test_loader, optimizers, num_epochs = 1, mode = 1)
                
                # print(test_change_precision_lst.shape)
                change_precision = test_change_precision_lst.reshape(prediction_window,1)*weights
                change_precision = change_precision.sum()/np.sum(weights)

                if epoch == 0:
                    test_change_precision_hist[:,0] = test_change_precision_lst
                else:
                    test_change_precision_hist = np.concatenate((test_change_precision_hist, test_change_precision_lst.reshape(prediction_window,1)), axis=1)
                eval_loss_hist.append(average_loss)
                if change_precision > best_prediction: 
                    has_improvement = True
                    print(f'\nNEW BEST prediction: {change_precision:.4f}%\n')
                    best_prediction = change_precision
                    best_model_state = model.state_dict()
                else:
                    print(f'\ncurrent change prediction precision: {change_precision:.4f}%\n')
            
                plt.clf()
                for i in range(prediction_window):
                    accuracies = test_change_precision_hist[i,:]
                    plt.plot(accuracies, label=f'{i+1} min accuracy', linestyle='solid')
                plt.plot(test_change_precision_hist.mean(axis=0), label=f'average accuracy', linestyle='dashed')
                plt.plot((test_change_precision_hist*weights).sum(axis=0)/np.sum(weights), label=f'weighted accuracy', linestyle='dotted')
                # plt.clf()
                # plt.plot(moving_average(eval_loss_hist, 3), label=f'loss', linestyle='solid')

                # actually train the model
                average_loss = work(model, train_loader, optimizers, test_every_x_epoch, mode = 0, schedulers = schedulers)
                # train_loss_hist.append(average_loss)
                # plt.plot(train_loss_hist, label=f'train loss', linestyle='dotted')
                if epoch == 0:
                    plt.legend() 
                    plt.pause(0.5)
                plt.pause(0.1)
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
        plt.show()
        plt.clf()
        # plt.ioff()

        encoder_lr = get_current_lr(encoder_optimizer)
        decoder_lr = get_current_lr(decoder_optimizer)
        lrs = [encoder_lr, decoder_lr]

        # Test the model
        # start_time = time.time()
        # print('Testing model')
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
        # print(f'testing completed in {time.time()-start_time:.2f} seconds')

        save_params(best_prediction, optimizers, model.state_dict(), best_model_state, model_path, last_model_path, model_training_param_path, has_improvement) 
        print('Normal exit. Model saved.')
        torch.cuda.empty_cache()
        gc.collect()
    except KeyboardInterrupt or Exception or TypeError:
        # save the model if the training was interrupted by keyboard input
        save_params(best_prediction, optimizers, model.state_dict(), best_model_state, model_path, last_model_path, model_training_param_path, has_improvement)
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