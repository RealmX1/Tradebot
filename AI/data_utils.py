import pandas as pd
import numpy as np
import torch
import time
import warnings
from torch.utils.data import Dataset, DataLoader # PyTorch data utilities

# if __name__ == '__main__': It seem no longer necessary to use this to import local file; I jsut have to add local path at the highest use case.
    
from normalization_param import *

def norm_param_2_idx(col_names):
    norm_param_2_idx_dict = {}
    # BB_mask = col_names.contains("BBL_|BBM_|BBU_").tolist() # bolinger band
    # MA_mask = col_names.contains("MA_").tolist() # moving average
    # CDL_mask = col_names.contains("open|high|low|close|vwap").tolist()
    # close_batch_mean_lst = [a or b or c for a, b, c in zip(BB_mask, MA_mask, CDL_mask)]
    # close_batch_mean_lst = np.where(close_batch_mean_lst)[0].tolist()
    # print("close_batch_mean_lst: ", close_batch_mean_lst, type(close_batch_mean_lst))
    # norm_param_2_idx_dict[NormParam.CloseBatch] = close_batch_mean_lst
    # print(norm_param_2_idx_dict[NormParam.CloseBatch])
    
    # volume_mask = col_names.contains("volume").tolist()
    # volume_index_lst = np.where(volume_mask)[0]
    # print("volume_index_lst: ", volume_index_lst, type(volume_index_lst))
    # norm_param_2_idx_dict[NormParam.Volume] = volume_index_lst
    # # how to normalize it? With respect to total share count (calculate turnover rate? 
    # # Or normalize it with respect to the average volume of the stock?
    # # both techniques result in some level of informaiton loss -- but proabably make the feature actually learnable.
    # trade_count_mask = col_names.contains("trade_count").tolist()
    # trade_count_index_lst = np.where(trade_count_mask)[0]
    # norm_param_2_idx_dict[NormParam.TradeCount] = trade_count_index_lst
    # # how to normalize it? probably just get share/trade using volume? Or should I drop it?
    # rsi_mask = col_names.contains("RSI").tolist()
    # rsi_index_lst = np.where(rsi_mask)[0]
    # norm_param_2_idx_dict[NormParam.RSI] = rsi_index_lst
    # cci_mask = col_names.contains("CCI").tolist()
    # cci_index_lst = np.where(cci_mask)[0]
    # norm_param_2_idx_dict[NormParam.CCI] = cci_index_lst
    # adx_mask = col_names.contains("ADX").tolist()
    # adx_index_lst = np.where(adx_mask)[0]
    # norm_param_2_idx_dict[NormParam.ADX] = adx_index_lst
    # dmp_mask = col_names.contains("DMP").tolist()
    # dmp_index_lst = np.where(dmp_mask)[0]
    # norm_param_2_idx_dict[NormParam.DMP] = dmp_index_lst
    # dmn_mask = col_names.contains("DMN").tolist()
    # dmn_index_lst = np.where(dmn_mask)[0]
    # norm_param_2_idx_dict[NormParam.DMN] = dmn_index_lst
    # dayofweek_mask = col_names.contains("dayofweek").tolist()
    # dayofweek_index_lst = np.where(dayofweek_mask)[0]
    # norm_param_2_idx_dict[NormParam.DayOfWeek] = dayofweek_index_lst
    # edt_scaled_mask = col_names.contains("edt_scaled").tolist()
    # edt_scaled_index_lst = np.where(edt_scaled_mask)[0]
    # norm_param_2_idx_dict[NormParam.EDT] = edt_scaled_index_lst
    # is_core_time_mask = col_names.contains("is_core_time").tolist()
    # is_core_time_index_lst = np.where(is_core_time_mask)[0]
    # norm_param_2_idx_dict[NormParam.IsCoreTime] = is_core_time_index_lst
    # cdl_mask = col_names.contains("CDL").tolist()
    # cdl_lst = np.where(cdl_mask)[0]
    # norm_param_2_idx_dict[NormParam.CDL] = cdl_lst
    # # TODO: can automate this part as well
    # # DONE!

    for x in NormParam:
        param = x.value
        norm_param = param['norm_param']
        search_str = param['search_str']
        # print("norm_param: ", norm_param, "search_str: ", search_str)
        mask = col_names.contains(search_str).tolist()
        idx_lst = np.where(mask)[0].tolist()
        norm_param_2_idx_dict[x] = idx_lst

    for key, val in norm_param_2_idx_dict.items():
        print(key, val)

    return norm_param_2_idx_dict # np2i_dict for short.

class StockDataset(Dataset):
    def __init__(self, data, prediction_window, not_close_batch_norm_lst, close_idx):
        print("not_close_batch_norm_lst: ", not_close_batch_norm_lst)

        self.close_idx = close_idx
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
        self.x_std = np.copy(self.x_mean)/1000 # actually not std here; it is just dividing close related features by mean close price.

        print("x_mean.shape: ", self.x_mean.shape)
        self.x_mean[:,not_close_batch_norm_lst] = 0 
        # NEED TO USE SIMILAR METHOD AS USED IN NORMALIZATION TO DETERMINE THE MEAN AND STD OF EACH FEATURE.


        self.x_std[:,not_close_batch_norm_lst] = 1
        # self.x_std[:,4:6] = 1
        # self.x_std[:,13:] = 1
        # print("x_mean.shape: ", self.x_mean.shape)
        # print("x.shape: ", self.x_raw.shape)
        # mean/std.shape: (data_num, feature_num)
        print("x_mean: ", self.x_mean)
        print("x_std: ", self.x_std)
        print("x_raw[0]: ", self.x_raw[0][0])
        self.x = (self.x_raw - self.x_mean[:,None,:])# / self.x_std[:,None,:]
        print("x[0]: ", self.x[0][0])
        # self.y = self.y - self.x_mean[:,0:1] # using this instead of self.y = self.y - self.x[:,-1,close_idx:close_idx+1]
        # doesn't make much sense. The target is to predict the the potential difference in value after last observed point.

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.x[idx,:,:], self.y[idx,:], self.x_raw[idx,-1,self.close_idx: self.close_idx+1] #self.x_mean[idx,:], self.x_std[idx,:]

    def get_item(self, idx):
        return self.__getitem__(idx)

class MultiStockDataset(Dataset):
    def __init__(self, data_lst, prediction_window, close_idx):
        self.dataset_lst = []
        for data in data_lst:
            self.dataset_lst.append(StockDataset(data, prediction_window, close_idx))
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

def sample_z_continuous(arr, z):
    n = arr.shape[0] - z + 1
    result = np.zeros((n, z, arr.shape[1]))
    for i in range(n):
        result[i] = arr[i:i+z]
    return result

# all things that need special normalization treatment are listed in np2i_dict & NormParam.py; 
# Default is direct standardization within scope of given input dataframe.

def normalize_data(df, np2i_dict):
    num_cols = df.shape[1] 
    data = df.values

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
    print(data_mean)
    print(data_std)
    

    # TODO: As feature num increase, it is becoming tedious to maintain mean&variance for special feature. Will need new structure for updating this in future.
    # DONE!

    data_norm = (data - data_mean) / data_std
    print("data_norm.shape: ", data_norm.shape)
    # print("normalized data: ", data_norm)
    return data_norm

def load_n_split_data(data_path, hist_window, prediction_window, batch_size, train_ratio, normalize = True, test = False):
    
    df = pd.read_csv(data_path, index_col = ['symbol', 'timestamp'])
    assert train_ratio < 1, "train_ratio should be less than 1"

    print("processing data")
    start_time = time.time()
    data_prep_window = hist_window + prediction_window
    
    col_names = df.columns.str
    col_num = df.shape[1]
    np2i_dict = norm_param_2_idx(col_names)

    if (normalize):
        data_norm = normalize_data(df, np2i_dict)
        data_norm = sample_z_continuous(data_norm, data_prep_window)

    train_size = int(train_ratio * len(df))
    val_size = len(df) - train_size
    train_data = data_norm[val_size:,:,:]
    val_data = data_norm[:val_size,:,:]

    col_idx_set = set(range(col_num))
    not_close_batch_norm_lst = list(col_idx_set - set(np2i_dict[NormParam.CloseBatch]))
    close_idx = df.columns.get_loc('close')
    if not test:
        train_dataset = StockDataset(train_data, prediction_window, not_close_batch_norm_lst, close_idx)
        val_dataset = StockDataset(val_data, prediction_window, not_close_batch_norm_lst, close_idx)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False) 
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
        # testing have shown that my gtx1080ti doesn't benefit from changing num_worker; but future hardware might need them.
        print(f'data loading completed in {time.time()-start_time:.2f} seconds')
        return train_loader, val_loader
    else:
        test_dataset = StockDataset(data_norm, prediction_window, not_close_batch_norm_lst, close_idx)
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
    else:
        data_norm = df.values
        test_dataset = TwoDStockDataset(data_norm, prediction_window, close_idx)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
        return test_loader, df.columns

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


def main():
    pass
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