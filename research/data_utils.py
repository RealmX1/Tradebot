import pandas as pd
import numpy as np
import torch
import time
from torch.utils.data import Dataset, DataLoader # PyTorch data utilities
from tsai.all import *

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
        y_raw = data[:,-prediction_window:,close_idx] 
        tmp = self.x_raw[:,-1,close_idx:close_idx+1]
        self.y = (y_raw - tmp)/tmp * 100 # don't need to normalize y; this is the best way; present target as the percentage growth with repsect to last close price.
        # print("y.shape: ",self.y.shape)
        # print("y:" , self.y)
        # y.shape: (data_num, output_size)
        self.x_mean = np.mean(self.x_raw[:,:,close_idx:close_idx+1], axis=1)
        self.x_mean = np.tile(self.x_mean, (1, feature_num))
        # self.x_std = np.std(self.x_raw[:,:,close_idx:close_idx+1], axis=1)
        # self.x_std = np.tile(self.x_std, (1, feature_num))


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

class TwoDStockDataset(Dataset):
    def __init__(self, data, prediction_window, close_idx):
        self.x = data
    
    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx,:], self.x[idx,close_idx], self.x[idx,close_idx] #self.x_mean[idx,:], self.x_std[idx,:]



def sample_z_continuous(arr, z):
    n = arr.shape[0] - z + 1
    result = np.zeros((n, z, arr.shape[1]))
    for i in range(n):
        result[i] = arr[i:i+z]
    return result

def normalize_data(df):
    # print(df.columns.tolist())
    BB_mask = df.columns.str.contains("BBL_|BBM_|BBU_").tolist() # bolinger band
    MA_mask = df.columns.str.contains("MA_").tolist() # moving average
    CDL_mask = df.columns.str.contains("open|high|low|close|vwap").tolist()
    close_batch_mean_list = [a or b or c for a, b, c in zip(BB_mask, MA_mask, CDL_mask)]
    # print(close_batch_mean_list)
    close_batch_mean_list = np.where(close_batch_mean_list)[0] # index 0 is needed to unpack return from np.where, which is a tuple of lists.
    # print(close_batch_mean_list)

    data = df.values
    
    zero_mean_one_var_mask = df.columns.str.contains("BBB_|BBP_|MACD").tolist()
    zero_mean_one_var_list = np.where(zero_mean_one_var_mask)[0]

    volume_mask = df.columns.str.contains("volume").tolist()
    volume_index = np.where(volume_mask)[0][0]
    trade_count_mask = df.columns.str.contains("trade_count").tolist()
    trade_count_index = np.where(trade_count_mask)[0][0]
    rsi_mask = df.columns.str.contains("RSI").tolist()
    rsi_index = np.where(rsi_mask)[0][0]
    cci_mask = df.columns.str.contains("CCI").tolist()
    cci_index = np.where(cci_mask)[0][0]
    adx_mask = df.columns.str.contains("ADX").tolist()
    adx_index = np.where(adx_mask)[0][0]
    dmp_mask = df.columns.str.contains("DMP").tolist()
    dmp_index = np.where(dmp_mask)[0][0]
    dmn_mask = df.columns.str.contains("DMN").tolist()
    dmn_index = np.where(dmn_mask)[0][0]
    dayofweek_mask = df.columns.str.contains("dayofweek").tolist()
    dayofweek_index = np.where(dayofweek_mask)[0][0]
    edt_scaled_mask = df.columns.str.contains("edt_scaled").tolist()
    edt_scaled_index = np.where(edt_scaled_mask)[0][0]
    is_core_time_mask = df.columns.str.contains("is_core_time").tolist()
    is_core_time_index = np.where(is_core_time_mask)[0][0]

    cdl_mask = df.columns.str.contains("CDL").tolist()
    cdl_list = np.where(cdl_mask)[0]
    print(cdl_list)

    
    data_mean = np.mean(data, axis = 0)
    data_std = np.std(data, axis = 0)
    # print(data_mean.shape)
    # data_mean[0:4] = data_mean[6:13] = 0 # data_mean[3] # use close mean for these columns
    data_mean[close_batch_mean_list] = 0
    data_mean[cdl_list] = 0
    data_mean[zero_mean_one_var_list] = 0
    zero_mean_one_var_list
    # data_mean[volume_index] = 0
    # data_mean[trade_count_index] = 0 # how should volume and trade_count be normalized?
    # volumn and trade_count should probably be normalized using 
    data_mean[rsi_index] = 50 # rsi_mean
    data_mean[cci_index] = 0 # cci_mean
    data_mean[adx_index] = 30.171159 # adx_mean
    data_mean[dmp_index] = 32.843816 # dmp_mean
    data_mean[dmn_index] = 32.276572 # dmn_mean
    data_mean[dayofweek_index] = 2   # day_of_week mean
    data_mean[edt_scaled_index] = 0.5 # edt_scaled
    data_mean[is_core_time_index] = 0.5 # is_core_time

    # how should mean for adx,dmp,and dmn be set?
    # print(data_mean)
    # data_std[0:4] = data_std[6:13] = 1 #data_std[3] # use close std for these columns
    data_std[close_batch_mean_list] = 1
    data_std[cdl_list] = 1
    data_std[zero_mean_one_var_list] = 1
    # data_std[volume_index] = 0
    # data_std[trade_count_index] = 0 # how should volume and trade_count be normalized?
    data_std[rsi_index] = 10 # rsi_std
    data_std[cci_index] = 100 # cci_std
    data_std[adx_index] = 16.460923 # adx_std
    data_std[dmp_index] = 18.971341 # dmp_std
    data_std[dmn_index] = 18.399032 # dmn_std
    data_std[dayofweek_index] = 1.414 # day_of_week std
    data_std[edt_scaled_index] = 1.25 # edt_scaled
    data_std[is_core_time_index] = 1 # is_core_time
    # # how should std for adx,dmp,and dmn be set?

    # As feature num increase, it is becoming tedious to maintain mean&variance for special feature. Will need new structure for updateing this in future.

    # print("data_mean shape: ", data_mean.shape)
    data_norm = (data - data_mean) / data_std
    print("normalized data: ", data_norm)
    return data_norm

def load_n_split_data(data_path, hist_window, prediction_window, batch_size, train_ratio, global_normalization_list = None, normalize = True):
    
    df = pd.read_csv(data_path, index_col = ['symbol', 'timestamp'])
    assert train_ratio < 1, "train_ratio should be less than 1"

    print("processing data")
    start_time = time.time()
    data_prep_window = hist_window + prediction_window

    if (normalize):
        data_norm = normalize_data(df)
        data_norm = sample_z_continuous(data_norm, data_prep_window)
    else:
        data_norm = df.values
        # data_norm = sample_z_continuous(data_norm, data_prep_window)
        test_dataset = TwoDStockDataset(data_norm, prediction_window, close_idx)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
        return test_loader, df.columns

    train_size = int(train_ratio * data_norm.shape[0])
    val_size = data_norm.shape[0] - train_size
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
        return test_loader, df.columns

def load_n_split_tsai_data(data_path, hist_window, prediction_window, batch_size, train_ratio, global_normalization_list = None, normalize = True):
    close_idx = 0
    print("processing data")
    start_time = time.time()
    
    data_prep_window = hist_window + prediction_window

    data = get_forecasting_time_series("Sunspots").values
    data_mean = np.mean(data, axis = 0)
    data_std = np.std(data, axis = 0)
    data_norm = (data-data_mean)/data_std
    print(data_norm[:5])
    
    print(data_norm.shape)

    data_norm = sample_z_continuous(data_norm, data_prep_window)
    print(data_norm.shape)

    train_size = int(train_ratio * data_norm.shape[0])
    val_size = data_norm.shape[0] - train_size
    train_data = data_norm[val_size:,:,:]
    val_data = data_norm[:val_size,:,:]

    train_dataset = StockDataset(train_data, prediction_window, close_idx)
    val_dataset = StockDataset(val_data, prediction_window, close_idx)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False) 
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    # testing have shown that my gtx1080ti doesn't benefit from changing num_worker; but future hardware might need them.
    print(f'data loading completed in {time.time()-start_time:.2f} seconds')
    return train_loader, val_loader
    

def prepare_data_alpaca(df, type = "bar"):
    # assumes that it is of 2 index levels: symbols and timestamps
    pass


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