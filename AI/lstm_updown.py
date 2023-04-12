import random
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

cwd = os.getcwd()
model_path = cwd+"/lstm_updown_S2S_bidirectional.pt"

# Load the CSV file into a Pandas dataframe
# time,open,high,low,close,Volume,Volume MA

close_idx = 3 # after removing time column


# Define hyperparameters
feature_num = input_size = 20 # Number of features (i.e. columns) in the CSV file -- the time feature is removed.
hidden_size = 1000    # Number of neurons in the hidden layer of the LSTM

num_layers  = 2     # Number of layers in the LSTM
output_size = 1     # Number of output values (closing price 1~10min from now)
prediction_window = 10
window_size = 32 # using how much data from the past to make prediction?
data_prep_window = window_size + prediction_window # +ouput_size becuase we need to keep 10 for calculating loss
drop_out = 0.1

learning_rate = 0.0001
batch_size = 1024

train_percent = 0.6

num_epochs = 10





# no_norm_num = 4 # the last four column of the data are 0s and 1s, no need to normalize them (and normalization might cause 0 division problem)

loss_fn = nn.MSELoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

plot_minutes = [0]


np.set_printoptions(precision=4, suppress=True)

# Define the LSTM model


    
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        # self.bn1 = nn.BatchNorm1d(1)
        
        assert encoder.hidden_size == decoder.hidden_size, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.num_layers == decoder.num_layers, \
            "Encoder and decoder must have equal number of layers!"

    # use teacher forcing ratio to balance between using predicted result vs. true result in generating next prediction
    def forward(self, input, target, teacher_forcing_ratio = 0.5):
        batch_size = input.shape[0]

        hidden, cell = self.encoder(input)
        # print("hidden.shape: ",hidden.shape)
        # expected: ?????(batch_size, hidden_size)

        outputs = torch.zeros(batch_size, prediction_window, output_size).to(self.device)
        x = target[:,0:1,None] # x at timestamp 

        for t in range (prediction_window):
            output, hidden, cell = self.decoder(x, hidden, cell)
            outputs[:,t:t+1,:] = output
            x = target[:,t:t+1,None] if random.random() < teacher_forcing_ratio else output
        
        return outputs.squeeze(2) # note that squeeze is used since y_batch is 2d, yet y_pred is 3d. (if output size sin't 1, then y_batch will be 3d.)            


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

def get_return_diff(x_batch,y_batch,y_pred):
    pass

def work(model, train_loader, optimizer, num_epochs = num_epochs, mode = 0): # mode 0: train, mode 1: test, mode 2: PLOT
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

    ma_predictions = 0
    ma_true_predictions = 0

    total_predictions = np.zeros(prediction_window) # one elemnt for each minute of prediction window
    total_true_predictions = np.zeros(prediction_window)

    average_loss = 0

    inverse_mask = torch.linspace(1, 11, 10)
    # print ("inverse_mask.shape: ", inverse_mask.shape)
    mask = torch.ones((prediction_window,))
    mask /= inverse_mask
    mask = mask.float().to(device)
    # print("mask: ", mask.device)
    # print("mask.shape: ", mask.shape)
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        ma_predictions *= 0.9
        ma_true_predictions *= 0.9
        i=0
        for i, (x_batch, y_batch, x_raw_close) in enumerate(train_loader):
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
            # y_pred *= mask
            # y_batch *= mask
            loss = loss_fn(y_pred, y_batch) 
            
            if mode == 0:
                optimizer.zero_grad() # removing zero_grad doesn't improve training speed (unlike some claimed); need more testing
                loss.backward()
                optimizer.step()

            total_cells, same_cells, total_cells_list,same_cells_list = get_direction_diff(x_batch, y_batch, y_pred)
            ma_predictions += total_cells
            ma_true_predictions += same_cells
            total_predictions += total_cells_list
            total_true_predictions += same_cells_list

            if mode == 2:
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
                else:
                    # predictions = torch.cat((predictions, prediction), dim=0)
                    raw_predictions = torch.cat((raw_predictions, raw_prediction), dim=0)
                    # targets = torch.cat((targets, target), dim=0)
                    raw_targets = torch.cat((raw_targets, raw_target), dim=0)
            
            epoch_loss += loss.item()
        epoch_loss /= (i+1)
        average_loss += epoch_loss
        correct_direction = ma_true_predictions / ma_predictions * 100

        # scheduler.step()
        print(f'Epoch {epoch+1:3}/{num_epochs:3}, Loss: {epoch_loss:10.7f}, Time per epoch: {(time.time()-start_time)/(epoch+1):.2f} seconds, Correct Direction: {correct_direction:.2f}%') # Learning Rate: {scheduler.get_last_lr()[0]:9.6f}
        
    print(f'completed in {time.time()-start_time:.2f} seconds')
    average_loss /= num_epochs
    accuracy_list = total_true_predictions / total_predictions * 100
    print("Accuracy List: ", accuracy_list)

    if mode == 2:
        return None,raw_predictions.cpu().numpy(), None, raw_targets.cpu().numpy()# return predictions.cpu().numpy(), raw_predictions.cpu().numpy(), targets.cpu().numpy(), raw_targets.cpu().numpy()
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

def main():
    start_time = time.time()
    print("loading data")
    # df = pd.read_csv('nvda_1min_complex_fixed.csv')
    # df = pd.read_csv("data/bar_set_huge_20180101_20230410_GOOG_indicator.csv", index_col = ['symbols', 'timestamps'])
    df = pd.read_csv("data/bar_set_huge_20180101_20230410_AAPL_indicator.csv", index_col = ['symbols', 'timestamps'])
    
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
    data_mean[15] = 50 # rsi_mean
    data_mean[16] = 0 # cci_mean
    data_mean[17] = 30.171159 # adx_mean
    data_mean[18] = 32.843816 # dmp_mean
    data_mean[19] = 32.276572 # dmn_mean
    # how should mean for adx,dmp,and dmn be set?
    # print(data_mean)
    data_std[0:4] = data_std[6:13] = 1 #data_std[3] # use close std for these columns
    data_std[15] = 10 # rsi_std
    data_std[16] = 100 # cci_std
    data_std[17] = 16.460923 # adx_std
    data_std[18] = 18.971341 # dmp_std
    data_std[19] = 18.399032 # dmn_std
    # how should std for adx,dmp,and dmn be set?
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
    total_loader = DataLoader(total_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    # testing have shown that my gtx1080ti doesn't benefit from changing num_worker; but future hardware might need them.
    print(f'data loading completed in {time.time()-start_time:.2f} seconds')


    print("loading model")
    start_time = time.time()
    encoder = Encoder(input_size, hidden_size, num_layers, drop_out).to(device)
    decoder = Decoder(output_size, hidden_size, num_layers, output_size, drop_out).to(device)
    model = Seq2Seq(encoder, decoder, device).to(device)
    if os.path.exists(model_path):
        print("Loading existing model")
        model.load_state_dict(torch.load(model_path))
    else:
        print("No existing model")
    print(model)
    print(f'model loading completed in {time.time()-start_time:.2f} seconds')

    # optimizer = SGD(model.parameters(), lr=learning_rate)
    optimizer = AdamW(model.parameters(),weight_decay=1e-5, lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=10, threshold=0.00005)

    

    try:
        # Train the model
        start_time = time.time()
        print("Training model")
        test_every_x_epoch = 1
        test_accuracy_hist = np.zeros((prediction_window,1))
        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}/{num_epochs}')
            if test_every_x_epoch and epoch % test_every_x_epoch == 0:
                test_accuracy_list, average_loss = work(model, test_loader, optimizer, num_epochs = 1, mode = 1)
                if epoch == 0:
                    test_accuracy_hist[:,0] = test_accuracy_list
                else:
                    test_accuracy_hist = np.concatenate((test_accuracy_hist, test_accuracy_list.reshape(prediction_window,1)), axis=1)
                # scheduler.step(average_loss)
                work(model, train_loader, optimizer, test_every_x_epoch, mode = 0)
        print(f'training completed in {time.time()-start_time:.2f} seconds')
        
        print(test_accuracy_hist.shape)
        # predictions = np.mean(predictions, axis = )
        # plt.ion()
        for i in range(prediction_window):
            predictions = test_accuracy_hist[i,:]
            plt.plot(predictions, label=f'{i+1} min prediction', linestyle='solid')
        plt.legend()
        print("showing?")
        plt.show()
        plt.clf()
        # plt.ioff()

        # Test the model
        start_time = time.time()
        print("Testing model")
        with torch.no_grad():
            work(model, test_loader, optimizer, 1, mode = 1)
        print(f'testing completed in {time.time()-start_time:.2f} seconds')


        # Make predictions
        start_time = time.time()
        print("Making Prediction")
        with torch.no_grad():
            predictions, raw_predictions, targets, raw_targets = work(model, total_loader, optimizer, 1, mode = 2)
            # plot(predictions, targets, test_size)
            plt.ioff()
            plot(raw_predictions, raw_targets, test_size)
        print(f'prediction completed in {time.time()-start_time:.2f} seconds')
                    

        torch.save(model.state_dict(), model_path)
        print('Normal exit. Model saved.')
    except KeyboardInterrupt or Exception or TypeError:
        # save the model if the training was interrupted by keyboard input
        # torch.save(model.state_dict(), model_path)
        print('Model not saved.')

if __name__ == '__main__':
    main()
    # cProfile.run('main()') # this shows execution time of each function. Might be useful for debugging & accelerating in detail.
