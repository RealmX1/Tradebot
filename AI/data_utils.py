import pandas as pd
import numpy as np
import torch
import time
import warnings
from torch.utils.data import Dataset, DataLoader # PyTorch data utilities

# if __name__ == '__main__': It seem no longer necessary to use this to import local file; I jsut have to add local path at the highest use case.
from model_structure_param import *
from normalization_param import *

def norm_param_2_idx(col_names):
    norm_param_2_idx_dict = {}

    for x in NormParam:
        param = x.value
        norm_param = param['norm_param']
        search_str = param['search_str']
        # print("norm_param: ", norm_param, "search_str: ", search_str)
        mask = col_names.contains(search_str).tolist()
        idx_lst = np.where(mask)[0].tolist()
        norm_param_2_idx_dict[x] = idx_lst

    # for key, val in norm_param_2_idx_dict.items(): # debug print
    #     print(key, val)

    return norm_param_2_idx_dict # np2i_dict for short.

def batch_norm(x_raw, not_close_batch_norm_lst, close_idx):
    x_mean = np.mean(x_raw[:,:,close_idx:close_idx+1], axis=1)
    x_mean = np.tile(x_mean, (1, feature_num))
    x_std = np.copy(x_mean) / 100
    x_mean[:,not_close_batch_norm_lst] = 0 
    x_std[:,not_close_batch_norm_lst] = 1

    return (x_raw - x_mean[:,None,:]) / x_std[:,None,:] # not a good idea to move the "/100" here... it should only apply to the close-related bathc


class StockDataset(Dataset):
    def __init__(self, data, timestamp, prediction_window, not_close_batch_norm_lst, close_idx): # assumes that first column of data is timestampstr.
        print('data.shape: ', data.shape)
        print('timestamp.shape: ', timestamp.shape)
        assert data.shape[0] == timestamp.shape[0], "data and timestamp must have the same number of rows."
        # print(data.shape)
        # print(timestamp.shape)
        # (N, hist_window+pred_window, feature_num)
        # (N, hist_window+pred_window, 1)
        
        self.timestamp = timestamp[:,-prediction_window-1] # timestamp of close price
        # (n) 1-d array
        # print("not_close_batch_norm_lst: ", not_close_batch_norm_lst)

        self.close_idx = close_idx
        feature_num = data.shape[2]
        x_raw = data[:,:-prediction_window,:] # slicing off the last entry of input
        # print("x.shape: ",self.x.shape)
        # x.shape: (data_num, window_size, feature_num)
        y_raw = data[:,-prediction_window:,close_idx] 
        tmp = x_raw[:,-1,close_idx:close_idx+1]
        self.y = (y_raw - tmp)/tmp * 100 * pct_pred_multiplier # don't need to normalize y; this is the best way; present target as the percentage growth with repsect to last close price.
        # print("y.shape: ",self.y.shape)
        # print("y:" , self.y)
        # y.shape: (data_num, output_size)
        # self.x_mean = np.mean(x_raw[:,:,close_idx:close_idx+1], axis=1)
        # self.x_mean = np.tile(self.x_mean, (1, feature_num))
        # # self.x_std = np.std(x_raw[:,:,close_idx:close_idx+1], axis=1)
        # # self.x_std = np.tile(self.x_std, (1, feature_num))
        # self.x_std = np.copy(self.x_mean)/1000 # actually not std here; it is just dividing close related features by mean close price.

        # self.x_mean[:,not_close_batch_norm_lst] = 0 
        # # NEED TO USE SIMILAR METHOD AS USED IN NORMALIZATION TO DETERMINE THE MEAN AND STD OF EACH FEATURE.


        # self.x_std[:,not_close_batch_norm_lst] = 1
        # # print("x_mean: ", np.mean(self.x_mean, axis=0))
        # # print("x_std: ", np.mean(self.x_std, axis=0))
        self.x = batch_norm(x_raw, not_close_batch_norm_lst, close_idx)

        print("x[0]: ", self.x[0][0])
        print('y[0]: ', self.y[0])
        
        print('timestamp: ', self.timestamp[0])
        # self.y = self.y - self.x_mean[:,0:1] # using this instead of self.y = self.y - self.x[:,-1,close_idx:close_idx+1]
        # doesn't make much sense. The target is to predict the the potential difference in value after last observed point.
        self.price = x_raw[:,-1,close_idx] # close price

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):

        return self.x[idx,:,:], self.y[idx,:], self.price[idx], self.timestamp[idx]  #self.x_mean[idx,:], self.x_std[idx,:]
        # (hist_window, feature_num), (prediction_window), (1), (1)
    def get_item(self, idx):
        return self.__getitem__(idx)

class MultiStockDataset(Dataset):
    def __init__(self, data_lst, prediction_window, not_close_batch_norm_lst, close_idx):
        self.dataset_lst = []
        for data, timestamp in data_lst:
            print("data.shape: ", data.shape)
            print("timestamp.shape: ", timestamp.shape)
            self.dataset_lst.append(StockDataset(data, timestamp, prediction_window, not_close_batch_norm_lst, close_idx))
        pass

    def __len__(self):
        return len(self.dataset_lst[0])

    def __getitem__(self, idx):
        x_lst = []
        y_lst = []
        x_raw_lst = []
        for data_set in self.dataset_lst:
            x, y, x_raw = data_set.get_item(idx)
            x_lst.append(x)
            y_lst.append(y)
            x_raw_lst.append(x_raw)

        return x_lst, y_lst, x_raw_lst

def sample_z_continuous(data, timestamp, z):
    n = data.shape[0] - z + 1
    feature_num = data.shape[1]
    result = np.zeros((n, z, feature_num))
    timestamp_lst = []

    for i in range(n):
        result[i] = data[i:i+z]
        timestamp_lst.append(timestamp[i:i+z])
    print("memory_usage: ", result.nbytes/(1024**2))
    # result = np.concatenate((result, time_stamp_result), axis=2)
    timestamp_result = np.array(timestamp_lst)
    print('timestamp_result.shape: ',timestamp_result.shape)
    return timestamp_result, result

# all things that need special normalization treatment are listed in np2i_dict & NormParam.py; 
# Default is direct standardization within scope of given input dataframe.

def normalize_data(df, np2i_dict): # takes df, return np.
    num_cols = df.shape[1] 
    data = df.values

    # timestamp_data = np.array(data[:,0:1], dtype='datetime64')
    timestamp_data = df.index.get_level_values('timestamp').to_numpy()
    # print('timestamp type: ', type(timestamp_data[0]))

    data_mean = np.mean(data, axis = 0)
    data_std = np.std(data, axis = 0)
    # print(data_mean.shape)
    # data_mean[0:4] = data_mean[6:13] = 0 # data_mean[3] # use close mean for these columns
    idx_lst = []
    for x in NormParam:
        idx_lst += np2i_dict[x]
        data_mean[np2i_dict[x]] = x.value['norm_param'][0]
        data_std[np2i_dict[x]] = x.value['norm_param'][1]
    
    assert len(set(idx_lst)) == len(idx_lst), print("There are duplicates in the list: Program isn't designed to handle this.")
    
    if num_cols < len(idx_lst):
        warnings.warn(f"{num_cols-len(idx_lst)} columns are directly standardized with respect to the scope of given dataframe.\n \
                      they are: {set(range(num_cols)) - set(idx_lst)}")
    # print('global mean: ', data_mean)
    # print('global std: ', data_std)

    data_norm = (data - data_mean) / data_std
    # print("data_norm.shape: ", data_norm.shape)
    # print("normalized data: ", data_norm)
    return timestamp_data, data_norm

def load_n_split_data(data_path, hist_window, prediction_window, batch_size, train_ratio, normalize = True, test = False):
    
    df = pd.read_csv(data_path, index_col = ['symbol', 'timestamp'])
    assert train_ratio < 1, "train_ratio should be less than 1"

    print("processing data")
    start_time = time.time()
    data_prep_window = hist_window + prediction_window
    
    col_names = df.columns.str
    col_num = df.shape[1]
    np2i_dict = norm_param_2_idx(col_names)

    # if (normalize):
    timestamp, data_norm = normalize_data(df, np2i_dict)
    timestamp, data_norm = sample_z_continuous(data_norm, timestamp, data_prep_window)

    train_size = int(train_ratio * len(df))
    val_size = len(df) - train_size
    train_data = data_norm[val_size:,:,:]
    train_timestamp = timestamp[val_size:,:]
    val_data = data_norm[:val_size,:,:]
    val_timestamp = timestamp[:val_size,:]

    col_idx_set = set(range(col_num))
    not_close_batch_norm_lst = list(col_idx_set - set(np2i_dict[NormParam.CloseBatch]))
    close_idx = df.columns.get_loc('close')
    if not test:
        train_dataset = StockDataset(train_data, train_timestamp, prediction_window, not_close_batch_norm_lst, close_idx)
        val_dataset = StockDataset(val_data, val_timestamp, prediction_window, not_close_batch_norm_lst, close_idx)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False) 
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
        # testing have shown that my gtx1080ti doesn't benefit from changing num_worker; but future hardware might need them.
        print(f'data loading completed in {time.time()-start_time:.2f} seconds')
        return train_loader, val_loader
    else:
        test_dataset = StockDataset(data_norm, timestamp, prediction_window, not_close_batch_norm_lst, close_idx)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
        print(f'data loading completed in {time.time()-start_time:.2f} seconds')
        return test_loader, df.columns

def load_multi_symbol_data(data_path_lst, hist_window, prediction_window, batch_size, train_ratio, close_idx, normalize = True, test = False):
    # assert train_ratio < 1, "train_ratio should be less than 1" # is code like this necessary?

    dfs = []
    for data_path in data_path_lst:
        df = pd.read_csv(data_path, index_col = ['symbol', 'timestamp'])
        dfs.append(df)

    print("processing data")
    start_time = time.time()
    data_prep_window = hist_window + prediction_window

    if (normalize):
        data_norm = normalize_data(df)
        data_norm = sample_z_continuous(data_norm, data_prep_window)
    # else:
    #     data_norm = df.values
    #     test_dataset = TwoDStockDataset(data_norm, prediction_window, close_idx)
    #     test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
    #     return test_loader, df.columns

    train_size = int(train_ratio * len(df))
    val_size = len(df) - train_size
    train_data = data_norm[val_size:,:,:]
    val_data = data_norm[:val_size,:,:]

    if not test:
        train_dataset = StockDataset(train_data, prediction_window, close_idx)
        val_dataset = StockDataset(val_data, prediction_window, close_idx)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False) 
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
        # testing have shown that my gtx1080ti doesn't benefit from changing num_worker; but future hardware might need them.
        print(f'data loading completed in {time.time()-start_time:.2f} seconds')
        return train_loader, val_loader
    else:
        test_dataset = StockDataset(data_norm, prediction_window, close_idx)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
        print(f'data loading completed in {time.time()-start_time:.2f} seconds')
        return test_loader, df.columns

def prepare_data_alpaca(df, type = "bar"):
    # assumes that it is of 2 index levels: symbols and timestamps
    pass

def load_randomized_training_set():

    return train_loader, val_loader

def main():
    tmp, tmp2 = load_n_split_data(training_data_path, hist_window, prediction_window, batch_size, train_ratio = 0.1, normalize = True, test = False)
    # data_path = "data/baba_test.csv"
    # df = pd.read_csv(data_path, index_col = ['symbol', 'timestamp'])
    # # open high low close volume trade_count vwap
    # # all other than trade_count will be normalized according to???
    # single_index_dfs = []
    # for symbol in df.index.levels[0]:
    #     single_index_df = df.loc[symbol].reset_index(level=0, drop=True)
    #     single_index_dfs.append(single_index_df)

    #     # self.x_mean = np.mean(self.x[:,:,close_idx:close_idx+1], axis=1)
    #     # self.x_std = np.std(self.x, axis=1)
    #     # self.x_mean[:,-no_norm_num:] = 0
    #     # self.x_std[:,-no_norm_num:] = 1 
    #     print(single_index_df.shape)
    # print(single_index_dfs[0])

if __name__ == "__main__":
    main()