import random
import json
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch # PyTorch
import torch.nn as nn # PyTorch neural network module
from torch.utils.data import Dataset, DataLoader # PyTorch data utilities
from torch.optim.lr_scheduler import StepLR # PyTorch learning rate scheduler
from torch.optim import AdamW, SGD
# from apex.optimizers import FusedLAMB

import matplotlib.pyplot as plt
import os
import numpy as np
import time
import cProfile


# import custom files
from S2S import *
from sim import *


policy = NaiveLong()
account = Account(100000, ['BABA'])

# Load the CSV file into a Pandas dataframe
# time,open,high,low,close,Volume,Volume MA

close_idx = 3 # after removing time column


# Define hyperparameters
feature_num = input_size = 23 # Number of features (i.e. columns) in the CSV file -- the time feature is removed.
hidden_size = 200    # Number of neurons in the hidden layer of the LSTM

num_layers  = 4     # Number of layers in the LSTM
output_size = 1     # Number of output values (closing price 1~10min from now)
prediction_window = 5
window_size = 60 # using how much data from the past to make prediction?
data_prep_window = window_size + prediction_window # +ouput_size becuase we need to keep 10 for calculating loss
dropout = 0.1

learning_rate = 0.0001
batch_size = 2000

train_percent = 0.5
num_epochs = 0





# no_norm_num = 4 # the last four column of the data are 0s and 1s, no need to normalize them (and normalization might cause 0 division problem)

loss_fn = nn.MSELoss(reduction = "none")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

plot_minutes = [0]


np.set_printoptions(precision=4, suppress=True)

# Define the LSTM model


    


# Define the dataset class
# data.shape: (data_num, data_prep_window, feature_num)
# SELF.Y IS ALREADY THE TRUE DIRECTION (SINCE LAST OBSERVED CLOSE)!!!
class NvidiaStockDataset(Dataset):
    def __init__(self, data):

        self.x_raw = data[:,:-prediction_window,:] # slicing off the last entry of input
        # print("x.shape: ",self.x.shape)
        # x.shape: (data_num, window_size, feature_num)
        self.y_raw = data[:,-prediction_window:,close_idx] 
        tmp = self.x_raw[:,-1,close_idx:close_idx+1]
        self.y = (self.y_raw - tmp)/tmp * 100 # don't need to normalize y; this is the best way; present target as the percentage growth with repsect to last close price.
        # print("y.shape: ",self.y.shape)
        # print("y:" , self.y)
        # y.shape: (data_num, output_size)
        self.x_mean = np.mean(self.x_raw[:,:,close_idx:close_idx+1], axis=1)
        self.x_mean = np.tile(self.x_mean, (1, feature_num))
        # print(self.x_mean)
        self.x_std = np.std(self.x_raw[:,:,close_idx:close_idx+1], axis=1)
        # print(self.x_std)
        self.x_std = np.tile(self.x_std, (1, feature_num))
        self.x_mean[:,4:6] = 0
        self.x_mean[:,13:] = 0
        # self.x_std[:,4:6] = 1
        # self.x_std[:,13:] = 1
        # print("x_mean.shape: ", self.x_mean.shape)
        # print("x.shape: ", self.x_raw.shape)
        # mean/std.shape: (data_num, feature_num)
        
        self.x = self.x_raw - self.x_mean[:,None,:] # / self.x_std[:,None,:]
        # self.y = self.y - self.x_mean[:,0:1] # using this instead of self.y = self.y - self.x[:,-1,close_idx:close_idx+1]
        # doesn't make much sense. The target is to predict the the potential difference in value after last observed point.

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.x[idx,:,:], self.y[idx,:], self.x_raw[idx,-1,close_idx:close_idx+1] #self.x_mean[idx,:], self.x_std[idx,:]

    # def get_raw(self, idx):
    #     return self.x_raw[idx,:,:], self.y_raw[idx,:], self.x_mean[idx,:], self.x_std[idx,:]


# Prepare the data
def sample_z_continuous(arr, z):
    n = arr.shape[0] - z + 1
    result = np.zeros((n, z, arr.shape[1]))
    for i in range(n):
        result[i] = arr[i:i+z]
    return result


def get_direction_diff(x_batch,y_batch,y_pred):
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


def calculate_policy_return(x_batch,y_batch,y_pred):
    pass

def work(model, train_loader, optimizers, num_epochs = num_epochs, mode = 0, schedulers = None): # mode 0: train, mode 1: test, mode 2: PLOT
    if mode == 0:
        teacher_forcing_ratio = 0.5
        model.train()
    else:
        teacher_forcing_ratio = 0
        model.eval()
    start_time = time.time()
    same_cells = 0
    
    predictions = None
    raw_predictions = None
    targets = None
    raw_targets = None
    raw_closes = None

    ma_predictions = 0
    ma_true_predictions = 0

    total_predictions = np.zeros(prediction_window) # one elemnt for each minute of prediction window
    total_true_predictions = np.zeros(prediction_window)

    average_loss = 0

    inverse_mask = torch.linspace(1, 11, 10)
    # print ("inverse_mask.shape: ", inverse_mask.shape)
    weights = torch.linspace(1, 0.1, steps=prediction_window).to(device)
    ma_loss = 0
    decisions = []
    buy_decisions = []
    sell_decisions = []
    account_value_hist = []
    price_hist = []
    for epoch in range(num_epochs):
        epoch_loss = 0
        ma_predictions *= 0.9
        ma_true_predictions *= 0.9
        i=0
        for i, (x_batch, y_batch, x_raw_close) in enumerate(train_loader):
            ma_loss *= 0.8
            # print("x_batch[0,:,:]: ", x_batch[0,:,:])
            # x_batch   [N, window_size, feature_num]
            # y_batch & output  [N, prediction_window]
            x_batch = x_batch.float().to(device) # probably need to do numpy to pt tensor in the dataset; need testing on efficiency #!!! CAN"T BE DONE. before dataloarder they are numpy array, not torch tensor
            # print("one input: ", x_batch[0:,:,:])
            y_batch = y_batch.float().to(device)
            
            y_pred = model(x_batch, y_batch, teacher_forcing_ratio) # [N, prediction_window]
            # print("y_pred.shape: ", y_pred.shape)
            # print("y_batch.shape: ", y_pred.shape)
            # print("y_batch: ", y_batch[0,:])
            loss = loss_fn(y_pred, y_batch)
            # print("loss shape: ", loss.shape)

            loss_val = loss.mean().item()

            weighted_loss = loss * weights
            final_loss = weighted_loss.mean()

            if ma_loss == 0:
                ma_loss = loss_val
            else:
                ma_loss += 0.2*loss_val
            
            if mode == 0:
                for optimizer in optimizers:
                    optimizer.zero_grad() # removing zero_grad doesn't improve training speed (unlike some claimed); need more testing
                final_loss.backward()
                for optimizer in optimizers:
                    optimizer.step()
                if schedulers is not None:
                    for scheduler in schedulers:
                        scheduler.step(final_loss)

            total_cells, same_cells, total_cells_list,same_cells_list = get_direction_diff(x_batch, y_batch, y_pred)
            ma_predictions += total_cells
            ma_true_predictions += same_cells
            total_predictions += total_cells_list
            total_true_predictions += same_cells_list

            if mode == 2:
                price = x_raw_close.item()
                decision = policy.decide('BABA', None, price, y_pred, account)
                decisions.append(decision)
                if decision[0] == 'b':
                    buy_decisions.append(1)
                    sell_decisions.append(0)
                elif decision[0] == 's':
                    buy_decisions.append(0)
                    sell_decisions.append(1)
                account_value_hist.append(account.evaluate())
                price_hist.append(price)
                # print(account.evaluate())
                # mean = mean.float().to(device)
                # std = std.float().to(device)
                
                raw_prediction = y_pred[:,plot_minutes]
                # prediction = raw_prediction * std[:,close_idx:close_idx+1] + mean[:,close_idx:close_idx+1]
                raw_target = y_batch[:,plot_minutes]
                # target = raw_target * std[:,close_idx:close_idx+1] + mean[:,close_idx:close_idx+1]

                
                if raw_predictions is None:
                    # predictions = prediction
                    raw_predictions = raw_prediction
                    # targets = target
                    raw_targets = raw_target
                    raw_closes = x_raw_close
                else:
                    # predictions = torch.cat((predictions, prediction), dim=0)
                    raw_predictions = torch.cat((raw_predictions, raw_prediction), dim=0)
                    # targets = torch.cat((targets, target), dim=0)
                    raw_targets = torch.cat((raw_targets, raw_target), dim=0)
                    raw_closes = torch.cat((raw_closes, x_raw_close), dim=0)
            
            epoch_loss += loss_val
        epoch_loss /= (i+1)
        average_loss += epoch_loss
        correct_direction = ma_true_predictions / ma_predictions * 100
            
        print(f'Epoch {epoch+1:3}/{num_epochs:3}, Loss: {epoch_loss:10.7f}, Time per epoch: {(time.time()-start_time)/(epoch+1):.2f} seconds, Correct Direction: {correct_direction:.2f}%, Encocder Learning Rate: {get_current_lr(optimizers[0]):9.10f}, Decoder Learning Rate: {get_current_lr(optimizers[1]):9.10f}') 
        
    print(f'completed in {time.time()-start_time:.2f} seconds')
    average_loss /= num_epochs
    accuracy_list = total_true_predictions / total_predictions * 100
    accuracy_list_print = [round(x, 2) for x in accuracy_list]
    print("Accuracy List: ", accuracy_list_print)

    if mode == 2:

        return buy_decisions, sell_decisions, account_value_hist, raw_predictions.cpu().numpy(), raw_targets.cpu().numpy(), price_hist # return predictions.cpu().numpy(), raw_predictions.cpu().numpy(), targets.cpu().numpy(), raw_targets.cpu().numpy()
    elif mode == 1:
        return accuracy_list, average_loss
    else:
        return average_loss

def plot(predictions, targets, test_size):
    # Plot the results
    print("total entry: ",predictions.shape[0])
    x = np.arange(len(predictions))
    print("predictions.shape: ",predictions.shape)
    plt.plot(targets, label='Actual')
    plt.plot(predictions, label='Predicted',linestyle='dotted')
    plt.legend()
    plt.axvline(x=test_size, color='r')
    plt.show()

def get_current_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def save_params(best_prediction, optimizers, model_state, best_model_state, model_path):
    print("saving params...")

    encoder_lr = get_current_lr(optimizers[0])
    decoder_lr = get_current_lr(optimizers[1])
    with open('training_param_log.json', 'w') as f:
        json.dump({'encoder_learning_rate': encoder_lr, 'decoder_learning_rate': decoder_lr, 'best_prediction': best_prediction}, f)
    print("saving model...")
    torch.save(best_model_state, model_path)
    torch.save(model_state, 'last'+model_path)
    print("done.")
    
def main():
    start_time = time.time()
    print("loading data")
    # "data/bar_set_huge_20180101_20230410_GOOG_indicator.csv"
    # "data/bar_set_huge_20200101_20230412_BABA_indicator.csv"
    data_path = "data/baba_test.csv"
    df = pd.read_csv(data_path, index_col = ['symbol', 'timestamp'])
    cwd = os.getcwd()
    model_path = 'lstm_updown_S2S_attention.pt'
    print("data loaded in ", time.time()-start_time, " seconds")
    
    # torch.backends.cudnn.benchmark = True # according to https://www.youtube.com/watch?v=9mS1fIYj1So, this speeds up cnn.
    print("processing data")
    start_time = time.time()
    data = df.values #.astype(float)
    # print("Raw data shape: ", data.shape)

    data_mean = np.mean(data, axis = 0)
    data_std = np.std(data, axis = 0)
    # print(data_mean.shape)
    data_mean[0:4] = data_mean[6:13] = 0 # data_mean[3] # use close mean for these columns
    # data_mean[15] = 50 # rsi_mean
    # data_mean[16] = 0 # cci_mean
    # data_mean[17] = 30.171159 # adx_mean
    # data_mean[18] = 32.843816 # dmp_mean
    # data_mean[19] = 32.276572 # dmn_mean
    # data_mean[20] = 2   # day_of_week mean
    # data_mean[21] = 0.5 # edt_scaled
    # data_mean[22] = 0.5 # is_core_time
    # # how should mean for adx,dmp,and dmn be set?
    # # print(data_mean)
    # data_std[0:4] = data_std[6:13] = 1 #data_std[3] # use close std for these columns
    # data_std[15] = 10 # rsi_std
    # data_std[16] = 100 # cci_std
    # data_std[17] = 16.460923 # adx_std
    # data_std[18] = 18.971341 # dmp_std
    # data_std[19] = 18.399032 # dmn_std
    # data_std[20] = 1.414 # day_of_week std
    # data_std[21] = 1.25 # edt_scaled
    # data_std[22] = 1 # is_core_time
    # # how should std for adx,dmp,and dmn be set?

    # As feature num increase, it is becoming tedious to maintain mean&variance for special feature. Will need new structure for updateing this in future.
    # 
    # print(data_std)


    data_std = (data - data_mean) / data_std

    data_std = sample_z_continuous(data_std, data_prep_window)
    print("Windowed data shape: ", data_std.shape)
    # (data_num, data_prep_window, output_size)

    train_size = int(len(data_std) * train_percent)
    test_size = int(len(data_std) * (1-train_percent))
    # learn on nearer data, and test on a previous data
    train_data = data_std[test_size:,:,:]
    test_data = data_std[:test_size,:,:]

    train_dataset = NvidiaStockDataset(train_data)
    test_dataset = NvidiaStockDataset(test_data)
    total_dataset = NvidiaStockDataset(data_std)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False) 
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    evaluation_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
    # testing have shown that my gtx1080ti doesn't benefit from changing num_worker; but future hardware might need them.
    print(f'data loading completed in {time.time()-start_time:.2f} seconds')


    print("loading model")
    start_time = time.time()
    model = Seq2Seq(input_size, hidden_size, num_layers, output_size, prediction_window, dropout, device).to(device)
    if os.path.exists('last'+model_path):
        print("Loading existing model")
        model.load_state_dict(torch.load('last'+model_path))
        with open('training_param_log.json', 'r') as f:
            saved_data = json.load(f)
            encoder_lr = saved_data['encoder_learning_rate']
            decoder_lr = saved_data['decoder_learning_rate']
            best_prediction = saved_data['best_prediction']
            start_best_prediction = best_prediction
    else:
        print("No existing model")
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
    encoder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(encoder_optimizer, mode='min', factor=0.98, patience=20, threshold=0.0001)
    decoder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(decoder_optimizer, mode='min', factor=0.98, patience=20, threshold=0.0001)

    optimizers = [encoder_optimizer, decoder_optimizer]
    schedulers = [encoder_scheduler, decoder_scheduler]
    

    try:
        plt.ion
        # Train the model
        start_time = time.time()
        print("Training model")
        test_every_x_epoch = 1
        test_accuracy_hist = np.zeros((prediction_window,1))
        weights = np.linspace(1, 0.1, num=prediction_window)
        weights = weights.reshape(prediction_window,1)

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
                    print(f'\nNEW BEST prediction: {accuracy:.2f}%\n')
                    best_prediction = accuracy
                    best_model_state = model.state_dict()
            
                plt.clf()
                for i in range(prediction_window):
                    accuracies = test_accuracy_hist[i,:]
                    plt.plot(accuracies, label=f'{i+1} min accuracy', linestyle='solid')
                plt.plot(test_accuracy_hist.mean(axis=0), label=f'average accuracy', linestyle='dashed')
                plt.plot((test_accuracy_hist*weights).sum(axis=0)/np.sum(weights), label=f'weighted accuracy', linestyle='dotted')
                if epoch == 0:
                    plt.legend()
                plt.pause(0.1)

                # actually train the model
                work(model, train_loader, optimizers, test_every_x_epoch, mode = 0, schedulers = schedulers)
        print(f'training completed in {time.time()-start_time:.2f} seconds')
        
        print("\n\n")
        if best_prediction > start_best_prediction:
            print(f'improved from {start_best_prediction:.2f}% to {best_prediction:.2f}%')
        else:
            print(f'NO IMPROVEMENET from {start_best_prediction:.2f}%')
        print("\n\n")

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

        print("Training Complete")
        plt.show()
        plt.clf()
        # plt.ioff()

        encoder_lr = get_current_lr(encoder_optimizer)
        decoder_lr = get_current_lr(decoder_optimizer)
        lrs = [encoder_lr, decoder_lr]

        # Test the model
        # start_time = time.time()
        # print("Testing model")
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


        # Make predictions
        start_time = time.time()
        print("Making Prediction")
        with torch.no_grad():
            buy_decisions, sell_decisions, account_value_hist, raw_predictions, raw_targets, price_hist = work(model, evaluation_loader, optimizers, num_epochs = 1, mode = 2)
            print(raw_predictions.shape)
            buy_time = [i for i, x in enumerate(buy_decisions) if x != 0]
            buy_price = [p for x, p in zip(buy_decisions,price_hist) if x != 0]

            sell_time = [i for i, x in enumerate(sell_decisions) if x != 0]
            sell_price = [p for x, p in zip(sell_decisions,price_hist) if x != 0]
            plt.plot(buy_time, buy_price, marker = '^', label = 'buy')
            plt.plot(sell_time, sell_price, marker = 'v', label = 'sell')


            plt.plot(account_value_hist, label='account value')
            plt.legend()
            # plot(predictions, targets, test_size)
            # plot(raw_predictions, raw_targets, test_size)
        print(f'prediction completed in {time.time()-start_time:.2f} seconds')
        save_params(best_prediction, optimizers, model.state_dict(), best_model_state, model_path) 
        print('Normal exit. Model saved.')
    except KeyboardInterrupt or Exception or TypeError:
        # save the model if the training was interrupted by keyboard input
        save_params(best_prediction, optimizers, model.state_dict(), best_model_state, model_path)

if __name__ == '__main__':
    main()
    # cProfile.run('main()') # this shows execution time of each function. Might be useful for debugging & accelerating in detail.
