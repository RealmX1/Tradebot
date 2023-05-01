import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch # PyTorch
import torch.nn as nn # PyTorch neural network module
from torch.utils.data import Dataset, DataLoader # PyTorch data utilities
from torch.optim.lr_scheduler import StepLR # PyTorch learning rate scheduler
from torch.optim import AdamW
# from apex.optimizers import FusedLAMB

import matplotlib.pyplot as plt
import os
import numpy as np
import time
import cProfile


close_idx = 3 # after removing time column

df = pd.read_csv('nvda_1min_complex_fixed.csv')

# Define hyperparameters
feature_num = input_size = 32 # Number of features (i.e. columns) in the CSV file -- the time feature is removed.
hidden_size = 32 # Number of neurons in the hidden layer of the LSTM

num_layers = 4 # Number of layers in the LSTM
output_size = 10 # Number of output values (closing price 1~10min from now)
learning_rate = 0.0001
num_epochs = 0
batch_size = 2048

window_size = 64 # using how much data from the past to make prediction?
data_prep_window = window_size + output_size # +ouput_size becuase we need to keep 10 for calculating loss
drop_out = 0.1

train_percent = 0.8

no_norm_num = 4 # the last four column of the data are 0s and 1s, no need to normalize them (and normalization might cause 0 division problem)

loss_fn = nn.MSELoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

plot_minutes = [0]



class NvidiaStockDataset(Dataset):
    def __init__(self, data):
        self.x = data[:,:-output_size,:] # slicing off the last entry of input
        # print("x.shape: ",self.x.shape)
        # x.shape: (data_num, window_size, feature_num)
        self.y = data[:,window_size:,close_idx] # moving the target entry one block forward
        # print("y.shape: ",self.y.shape)
        # y.shape: (data_num, output_size)
        self.x_mean = np.mean(self.x, axis=1)
        self.x_std = np.std(self.x, axis=1)
        self.x_mean[:,-no_norm_num:] = 0
        self.x_std[:,-no_norm_num:] = 1 
        print("x_mean.shape: ", self.x_mean.shape)

        close_mean = self.x_mean[:,close_idx:close_idx+1]
        close_mean = np.tile(close_mean, (1, 13))

        # mean/std.shape: (data_num, feature_num)

        # does this normalization broadcast work properly? 
        # desired effect is x[i,:,j] will be normalized using x_mean[i,j] and x_std[i,j],
        # and y[i,j] will be normalized using x_mean[i,close_idx] and x_std[i,close_idx]
        self.x = (self.x - self.x_mean[:,None,:]) / self.x_std[:,None,:]
        self.y = (self.y - self.x_mean[:,close_idx:close_idx+1]) / self.x_std[:,close_idx:close_idx+1]

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.x[idx,:,:], self.y[idx,:], self.x_mean[idx,:], self.x_std[idx,:]


# Prepare the data
def sample_z_continuous(arr, z):
    n = arr.shape[0] - z + 1
    result = np.zeros((n, z, arr.shape[1]))
    for i in range(n):
        result[i] = arr[i:i+z]
    return result


df_float = df.values #.astype(float)
df_float = df_float[:,1:] # remove the utftime column.
# change the utftime column to a column of day and a time of tradingday instead?

# data_num = df_float.shape[0]
print("df_float shape:",df_float.shape)
# (data_num, output_size)

# am I doing this correctly? Should LSTM be trained this way?
# or should it be trained using continuous dataset, and progress by feeding one data after another?
# at least current method makes it easier to noramlize each window of input independently.
data = result = sample_z_continuous(df_float, data_prep_window)
print(data.shape)
# (data_num, data_prep_window, output_size)

train_size = int(len(data) * train_percent)
test_size = int(len(data) * (1-train_percent))
# learn on nearer data, and test on a previous data; 
# not sure which order is better... don't have knowledge of such metric; probably should do experiment and read paper on this
# train_data = data[:train_size,:,:]
# test_data = data[train_size:,:,:]
train_data = data[test_size:,:,:]
test_data = data[:test_size,:,:]

train_dataset = NvidiaStockDataset(train_data)
test_dataset = NvidiaStockDataset(test_data)
total_dataset = NvidiaStockDataset(data)

print(df.columns.tolist())

np.set_printoptions(precision=2, suppress=True, threshold=np.inf)
# for x in range(1):
#     print("training sequence = ", train_dataset[x][0])


