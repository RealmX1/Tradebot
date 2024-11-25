import random
import json
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
np.set_printoptions(precision=4, suppress=True) 


# import custom files
from S2S import *
from sim import *
from data_utils import * 


# policy = NaiveLong()
# account = Account(100000, ['AAPL'])

close_idx = 3 # after removing time column

# Define hyperparameters
feature_num         = input_size = 23  # candel  # Number of features (i.e. columns) in the CSV file -- the time feature is removed.
hidden_size         = 200    # Number of neurons in the hidden layer of the LSTM
num_layers          = 1    # Number of layers in the LSTM
output_size         = 1     # Number of output values (closing price 1~10min from now)
prediction_window   = 2
hist_window         = 30 # using how much data from the past to make prediction?
data_prep_window    = hist_window + prediction_window # +ouput_size becuase we need to keep 10 for calculating loss


learning_rate   = 0.0001
batch_size      = 10000
train_ratio     = 0.9
num_epochs      = 20
dropout         = 0.1

loss_fn = nn.MSELoss(reduction = 'none')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

plot_minutes = [0]


torch.autograd.set_detect_anomaly(True)


def get_direction_diff(y_batch,y_pred):
    # start_time = time.time()
    # print('start get_direction_diff')
    # true_direction = y_batch-x_batch[:,-1,close_idx:close_idx+1]

    true_direction = np.clip(y_batch.cpu().numpy(), 0, np.inf) # this turns negative to 0, positive to 1
    true_direction[true_direction != 0] = 1
    # true_direction = y_batch.cpu().numpy()


    pred_direction = np.clip(y_pred.clone().detach().cpu().numpy(), 0, np.inf) # turn all 
    pred_direction[pred_direction != 0] = 1
    # pred_direction[pred_direction == 0.5] = 0
    # pred_direction = y_pred.clone().detach().cpu().numpy()

    # print('True: ', true_direction.shape)
    # print('Pred: ', pred_direction)

    instance_num =  true_direction.shape[0]
    prediction_min = true_direction.shape[1]

    total_cells = instance_num * prediction_min
    same_cells = np.count_nonzero(true_direction == pred_direction)

    total_cells_list = np.full((prediction_min,), instance_num)
    same_cells_list = np.count_nonzero(true_direction == pred_direction, axis = 0)
    # print('total_cells: ',total_cells)
    # print('same_cells.shape: ',same_cells.shape)
    # print(type(true_direction))
    # print(true_direction.typedf())
    tp = np.sum((true_direction == 1) & (pred_direction == 1))
    fp = np.sum((true_direction == 0) & (pred_direction == 1))

    total_up = np.sum(true_direction == 1)
    total_down = np.sum(true_direction == 0)

    # print('get_direction_diff time: ', time.time()-start_time)
    return total_cells, same_cells, total_cells_list, same_cells_list, tp, fp, total_up, total_down


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


    total_predictions = np.zeros(prediction_window) # one elemnt for each minute of prediction window
    total_true_predictions = np.zeros(prediction_window)

    average_loss = 0

    inverse_mask = torch.linspace(1, 11, 10)
    # print ('inverse_mask.shape: ', inverse_mask.shape)
    weight_decay = 0.2
    weights = torch.pow(torch.tensor(weight_decay), torch.arange(prediction_window).float()).to(device)
    # ([1.0000, 0.8000, 0.6400, 0.5120, 0.4096, 0.3277, 0.2621, 0.2097, 0.1678,0.1342])
    # weights = torch.linspace(1, 0.1, steps=prediction_window)
    # ma_loss = None
    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_predictions = 0
        epoch_true_predictions = 0
        epoch_tp = 0
        epoch_fp = 0
        epoch_up = 0
        epoch_down = 0
        i=0
        for i, (x_batch, y_batch, x_raw_close) in enumerate(data_loader):
            
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

            total_cells, same_cells, total_cells_list,same_cells_list, tp, fp, up, down = get_direction_diff(y_batch, y_pred)
            epoch_predictions += total_cells
            epoch_true_predictions += same_cells
            epoch_tp += tp
            epoch_fp += fp
            epoch_up += up
            epoch_down += down
            total_predictions += total_cells_list
            total_true_predictions += same_cells_list

            
            epoch_loss += loss_val
        epoch_loss /= (i+1)
        average_loss += epoch_loss
        accuracy = epoch_true_predictions / epoch_predictions * 100
            
        print(f'Epoch {epoch+1:3}/{num_epochs:3}, ' +
              f'Loss: {epoch_loss:10.7f}, ' +
              f'Time per epoch: {(time.time()-start_time)/(epoch+1):.2f} seconds, ' +
              f'Correct Direction: {accuracy:.2f}%, ' +
              f'Encocder LR: {get_current_lr(optimizers[0]):9.6f}, Decoder LR: {get_current_lr(optimizers[1]):9.6f}, ' +
              f'Precision: {epoch_tp/(epoch_tp+epoch_fp)*100:7.4f}%,' +
              f'\nBackground down percent: {epoch_up/(epoch_up+epoch_down)*100:7.4f}%')
              # f'Weighted Loss: {final_loss.item():10.7f}, MA Loss: {ma_loss.mean().item():10.7f}') 
            
    print(f'completed in {time.time()-start_time:.2f} seconds')
    average_loss /= num_epochs
    accuracy_list = total_true_predictions / total_predictions * 100
    accuracy_list_print = [round(x, 3) for x in accuracy_list]
    print('Accuracy List: ', accuracy_list_print)

    if mode == 1:
        return accuracy_list, average_loss
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

def save_params(best_prediction, optimizers, model_state, best_model_state, model_path, last_model_path, model_param_path):
    print('saving params...')

    encoder_lr = get_current_lr(optimizers[0])
    decoder_lr = get_current_lr(optimizers[1])
    with open(model_param_path, 'w') as f:
        json.dump({'encoder_learning_rate': encoder_lr, 'decoder_learning_rate': decoder_lr, 'best_prediction': best_prediction}, f)
    print('saving model...')
    torch.save(best_model_state, model_path)
    torch.save(model_state, last_model_path)
    print('done.')
    
def main():
    # CHANGE CONFIG NAME to save a new model
    config_name = 'lstm_updown_S2S_attention_23feature'
    # torch.cuda.empty_cache()
    # gc.collect()
    # torch.backends.cudnn.benchmark = True # according to https://www.youtube.com/watch?v=9mS1fIYj1So, this speeds up cnn.
    
    start_time = time.time()
    print('loading data & model')
    # 'data/bar_set_huge_20180101_20230410_GOOG_indicator.csv'
    # 'data/bar_set_huge_20200101_20230412_BABA_indicator.csv'
    # data_path = 'data/cdl_test_2.csv'
    data_path = '../data/csv/bar_set_huge_20200101_20230417_AAPL_indicator.csv'
    model_path = f'../model/model_{config_name}.pt'
    last_model_path = f'../model/last_model_{config_name}.pt'
    model_param_path = f'../model/training_param_{config_name}.json'
    print('loaded in ', time.time()-start_time, ' seconds')
    
    train_loader, test_loader = load_n_split_data(data_path, hist_window, prediction_window, batch_size, train_ratio, global_normalization_list = None)
    


    print('loading model')
    start_time = time.time()
    model = Seq2Seq(input_size, hidden_size, num_layers, output_size, prediction_window, dropout, device, attention = True).to(device)
    if os.path.exists(last_model_path):
        print('Loading existing model')
        model.load_state_dict(torch.load(last_model_path))
        with open(model_param_path, 'r') as f:
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
    best_model_state = model.state_dict()

    print(model)
    print(f'model loading completed in {time.time()-start_time:.2f} seconds')


    # optimizer = SGD(model.parameters(), lr=learning_rate)
    encoder_optimizer = AdamW(model.encoder.parameters(),weight_decay=1e-5, lr=encoder_lr)
    decoder_optimizer = AdamW(model.decoder.parameters(),weight_decay=1e-5, lr=decoder_lr)
    encoder_scheduler = ReduceLROnPlateau(encoder_optimizer, mode='min', factor=0.95, patience=10, threshold=0.0001)
    decoder_scheduler = ReduceLROnPlateau(decoder_optimizer, mode='min', factor=0.95, patience=10, threshold=0.0001)

    optimizers = [encoder_optimizer, decoder_optimizer]
    schedulers = [encoder_scheduler, decoder_scheduler]
    

    try:
        plt.ion
        # Train the model
        start_time = time.time()
        print('Training model')
        test_every_x_epoch = 1
        test_accuracy_hist = np.zeros((prediction_window,1))

        weight_decay = 0.2
        arr = np.ones(prediction_window)
        for i in range(1, prediction_window):
            arr[i] = arr[i-1] * weight_decay
        weights = arr.reshape(prediction_window,1)

        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}/{num_epochs}')
            if test_every_x_epoch and epoch % test_every_x_epoch == 0:
                test_accuracy_list, average_loss = work(model, test_loader, optimizers, num_epochs = 1, mode = 1)
                
                accuracy = test_accuracy_list.reshape(prediction_window,1)*weights
                accuracy = accuracy.sum()/np.sum(weights)

                if epoch == 0:
                    test_accuracy_hist[:,0] = test_accuracy_list
                else:
                    test_accuracy_hist = np.concatenate((test_accuracy_hist, test_accuracy_list.reshape(prediction_window,1)), axis=1)
                
                if accuracy > best_prediction: 
                    print(f'\nNEW BEST prediction: {accuracy:.4f}%\n')
                    best_prediction = accuracy
                    best_model_state = model.state_dict()
                else:
                    print(f'\ncurrent prediction: {accuracy:.4f}%\n')
            
                # plt.clf()
                for i in range(prediction_window):
                    accuracies = test_accuracy_hist[i,:]
                    plt.plot(accuracies, label=f'{i+1} min accuracy', linestyle='solid')
                plt.plot(test_accuracy_hist.mean(axis=0), label=f'average accuracy', linestyle='dashed')
                plt.plot((test_accuracy_hist*weights).sum(axis=0)/np.sum(weights), label=f'weighted accuracy', linestyle='dotted')
                if epoch == 0:
                    plt.legend() 
                    plt.pause(0.5)
                plt.pause(0.1)

                # actually train the model
                work(model, train_loader, optimizers, test_every_x_epoch, mode = 0, schedulers = schedulers)
        print(f'training completed in {time.time()-start_time:.2f} seconds')
        
        print('\n\n')
        if best_prediction > start_best_prediction:
            print(f'improved from {start_best_prediction:.2f}% to {best_prediction:.2f}%')
        else:
            print(f'NO IMPROVEMENET from {start_best_prediction:.2f}%')
        print('\n\n')

        print(test_accuracy_hist.shape)
        # predictions = np.mean(predictions, axis = )
        # plt.ion()
        # for i in range(prediction_window):
        #     predictions = test_accuracy_hist[i,:]
        #     plt.plot(predictions, label=f'{i+1} min prediction', linestyle='solid')
        # plt.plot(test_accuracy_hist.mean(axis=0), label=f'average prediction', linestyle='dashed')
        # weights = np.linspace(1, 0.1, num=prediction_window)
        # weights = weights.reshape(prediction_window,1)
        # plt.plot((test_accuracy_hist*weights).sum(axis=0)/np.sum(weights), label=f'weighted prediction', linestyle='dotted')
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
        #     test_accuracy_list, average_loss = work(model, test_loader, optimizers, num_epochs = 1, mode = 1)
        #     test_accuracy_hist[:,0] = test_accuracy_list
        #     plt.clf()
        #     for i in range(prediction_window):
        #         predictions = test_accuracy_hist[i,:]
        #         plt.plot(predictions, 0, label=f'{i+1} min accuracy', marker = 'o')
        #     plt.plot(test_accuracy_hist.mean(axis=0), 0, label=f'average accuracy', marker = 'o')
        #     plt.plot((test_accuracy_hist*weights).sum(axis=0)/np.sum(weights), label=f'weighted accuracy', linestyle='dotted')
        #     plt.legend()
        #     plt.ylim(0, 1)
        #     plt.show()
        #     plt.pause(1)
        # print(f'testing completed in {time.time()-start_time:.2f} seconds')

        save_params(best_prediction, optimizers, model.state_dict(), best_model_state, model_path, last_model_path, model_param_path) 
        print('Normal exit. Model saved.')
        torch.cuda.empty_cache()
        gc.collect()
    except KeyboardInterrupt or Exception or TypeError:
        # save the model if the training was interrupted by keyboard input
        save_params(best_prediction, optimizers, model.state_dict(), best_model_state, model_path, last_model_path, model_param_path)
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == '__main__':
    main()
    # cProfile.run('main()') # this shows execution time of each function. Might be useful for debugging & accelerating in detail.
