import pandas as pd
import numpy as np
import torch
import time
from torch.utils.data import Dataset, DataLoader # PyTorch data utilities

close_idx = 3

# Define the dataset class
# data.shape: (data_num, data_prep_window, feature_num)
# SELF.Y IS ALREADY THE TRUE DIRECTION (SINCE LAST OBSERVED CLOSE)!!!
class StockDataset(Dataset):
    def __init__(self, data, prediction_window, close_idx):
        feature_num = data.shape[2]
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
        self.x_std = np.std(self.x_raw[:,:,close_idx:close_idx+1], axis=1)
        self.x_std = np.tile(self.x_std, (1, feature_num))

        # non "close" related indicators are not normalized again for each data sample
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

def sample_z_continuous(arr, z):
    n = arr.shape[0] - z + 1
    result = np.zeros((n, z, arr.shape[1]))
    for i in range(n):
        result[i] = arr[i:i+z]
    return result

def normalize_data(df):
    data = df.values
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
    # how should mean for adx,dmp,and dmn be set?
    # print(data_mean)
    data_std[0:4] = data_std[6:13] = 1 #data_std[3] # use close std for these columns
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

    # print("data_mean shape: ", data_mean.shape)
    data_norm = (data - data_mean) / data_std
    print("normalized data: ", data_norm)
    return data_norm

def load_n_split_data(data_path, hist_window, prediction_window, batch_size, train_ratio, global_normalization_list = None):
    
    df = pd.read_csv(data_path, index_col = ['symbol', 'timestamp'])
    assert train_ratio < 1, "train_ratio should be less than 1"

    print("processing data")
    start_time = time.time()
    data_prep_window = hist_window + prediction_window

    data_norm = normalize_data(df)
    
    data_norm = sample_z_continuous(data_norm, data_prep_window)

    train_size = int(train_ratio * len(df))
    val_size = len(df) - train_size
    train_data = data_norm[val_size:,:,:]
    val_data = data_norm[:val_size,:,:]

    if (train_ratio != 0):
        train_dataset = StockDataset(train_data, prediction_window, close_idx)
        val_dataset = StockDataset(val_data, prediction_window, close_idx)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False) 
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
        # testing have shown that my gtx1080ti doesn't benefit from changing num_worker; but future hardware might need them.
        print(f'data loading completed in {time.time()-start_time:.2f} seconds')
        return train_loader, val_loader
    else: # if train_ratio is 0, only return step by step
        test_dataset = StockDataset(data_norm, prediction_window, close_idx)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
        print(f'data loading completed in {time.time()-start_time:.2f} seconds')
        return test_loader



def prepare_data_alpaca(df, type = "bar"):
    # assumes that it is of 2 index levels: symbols and timestamps
    df  


def main():
    data_path = "data/baba_test.csv"
    df = pd.read_csv(data_path, index_col = ['symbol', 'timestamp'])
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