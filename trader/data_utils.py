import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch # PyTorch
import numpy as np
from torch.utils.data import Dataset

close_idx = 3 # after removing time column

def sample_z_continuous(arr, z):
    n = arr.shape[0] - z + 1
    result = np.zeros((n, z, arr.shape[1]))
    for i in range(n):
        result[i] = arr[i:i+z]
    return result

class NvidiaStockDataset(Dataset):
    def __init__(self, data):
        self.x = data[:,:-1,:] # slicing off the last entry of input
        self.y = data[:,1:,close_idx:close_idx+1] # moving the target entry one block forward
        # print(self.x.shape)
        # print(self.y.shape)
        assert self.x.shape[0] == self.y.shape[0]

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.x[idx,:,:], self.y[idx,:,:]