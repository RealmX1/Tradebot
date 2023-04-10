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


def prepare_data_trading_view(train_percent=0.8, data_prep_window=65):
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




def prepare_data_alpaca(df, type = "bar"):
    # assumes that it is of 2 index levels: symbols and timestamps
    df  

def main():
    df = pd.read_csv('raw_from_data.csv', index_col = ['symbols', 'timestamps'])
    # open high low close volume trade_count vwap
    # all other than trade_count will be normalized according to???
    single_index_dfs = []
    for symbol in df.index.levels[0]:
        single_index_df = df.loc[symbol].reset_index(level=0, drop=True)
        single_index_dfs.append(single_index_df)

        # self.x_mean = np.mean(self.x[:,:,close_idx:close_idx+1], axis=1)
        # self.x_std = np.std(self.x, axis=1)
        # self.x_mean[:,-no_norm_num:] = 0
        # self.x_std[:,-no_norm_num:] = 1 
        print(single_index_df.shape)
    print(single_index_dfs[0])

if __name__ == "__main__":
    main()