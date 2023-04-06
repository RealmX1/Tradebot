import pandas as pd
import numpy as np
from torch.utils.data import Dataset

close_idx = 3

class NvidiaStockDataset(Dataset):
    def __init__(self, data):
        self.x = data[:, :-1, :]
        self.y = data[:, 1:, close_idx:close_idx+1]
        assert self.x.shape[0] == self.y.shape[0]

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.x[idx, :, :], self.y[idx, :, :]


def sample_z_continuous(arr, z):
    n = arr.shape[0] - z + 1
    result = np.zeros((n, z, arr.shape[1]))
    for i in range(n):
        result[i] = arr[i:i+z]
    return result


def prepare_data(train_percent=0.8, data_prep_window=65):
    df = pd.read_csv('nvda_1min.csv')

    df_float = df.values.astype(float)
    df_float = df_float[:,1:]
    feature_means = np.mean(df_float, axis=0,keepdims=True)
    close_mean = feature_means[0, close_idx]
    feature_stds = np.std(df_float, axis=0,keepdims=True)
    close_stds = feature_stds[0, close_idx]

    df_float = (df_float - feature_means) / feature_stds
    data = result = sample_z_continuous(df_float, data_prep_window)

    train_size = int(len(data) * train_percent)
    test_size = int(len(data) * (1 - train_percent))
    train_data = data[test_size:, :, :]
    test_data = data[:test_size, :, :]

    train_dataset = NvidiaStockDataset(train_data)
    test_dataset = NvidiaStockDataset(test_data)
    total_dataset = NvidiaStockDataset(data)

    return train_dataset, test_dataset, total_dataset
    