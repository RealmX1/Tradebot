from enum import Enum
class NormParam(Enum):
    CloseBatch  = {'norm_param': (0,1), 'search_str': 'BBL_|BBM_|BBU_|MA_|open|high|low|close|vwap'}
    NoNorm      = {'norm_param': (0,1), 'search_str': 'BBB_|BBP_|MACD'}
    Volume      = {'norm_param': (0,1), 'search_str': 'volume'}
    TradeCount  = {'norm_param': (0,1), 'search_str': 'trade_count'}
    RSI         = {'norm_param': (50,10), 'search_str': 'RSI'}
    CCI         = {'norm_param': (0,100), 'search_str': 'CCI'}
    ADX         = {'norm_param': (30.171159,16.460923), 'search_str': 'ADX'}
    DMP         = {'norm_param': (32.843816,18.971341), 'search_str': 'DMP'}
    DMN         = {'norm_param': (32.276572,18.399032), 'search_str': 'DMN'}
    DayOfWeek   = {'norm_param': (2,1.414), 'search_str': 'dayofweek'}
    EDT         = {'norm_param': (0.5,1.25), 'search_str': 'edt_scaled'}
    IsCoreTime  = {'norm_param': (0.5,1), 'search_str': 'is_core_time'}
    CDL         = {'norm_param': (0,1), 'search_str': 'CDL'}
    # how should std for adx,dmp,and dmn be set?

    # volume might need to be normalized with respect to the stocks' own volume (i.e., no need for special treatment)
    # or it can be normalized with respect to total volume of the stock.. I'm not sure. 
    # if volume is to be normalized according to total volume, do it in indicators.py step, not in data_util.
    # also, is alpaca's volume data adjusted for split? if not, it is pretty much useless without intense processing

    # trade_count should probably be normalized using volume/trade. (i beleive this is the only useful information that can be gained from trade count)
    
    # how should mean for adx,dmp,and dmn be set? Are they calculated irrelavent to relative price of the stock?




    # BB_mask = col_names.contains("BBL_|BBM_|BBU_").tolist() # bolinger band
    # MA_mask = col_names.contains("MA_").tolist() # moving average
    # CDL_mask = col_names.contains("open|high|low|close|vwap").tolist()
    # close_batch_mean_lst = [a or b or c for a, b, c in zip(BB_mask, MA_mask, CDL_mask)]
    # close_batch_mean_lst = np.where(close_batch_mean_lst)[0].tolist()
    # print("close_batch_mean_lst: ", close_batch_mean_lst, type(close_batch_mean_lst))
    # norm_param_2_idx_dict[NormParam.CloseBatch] = close_batch_mean_lst
    # print(norm_param_2_idx_dict[NormParam.CloseBatch])



















# the following code isn't actually directly useful here. I brought it here to test the github copilot's auto completion capability "QiangDa"
# it worked much better than expected.

# data = df.values

# data_mean = np.mean(data, axis = 0)
# data_std = np.std(data, axis = 0)
# # print(data_mean.shape)
# # data_mean[0:4] = data_mean[6:13] = 0 # data_mean[3] # use close mean for these columns
# data_mean[close_batch_mean_list] = 0
# data_mean[cdl_list] = 0
# data_mean[zero_mean_one_var_list] = 0
# zero_mean_one_var_list
# # data_mean[volume_index] = 0
# # data_mean[trade_count_index] = 0 # how should volume and trade_count be normalized?
# # volumn and trade_count should probably be normalized using 
# data_mean[rsi_index] = 50 # rsi_mean
# data_mean[cci_index] = 0 # cci_mean
# data_mean[adx_index] = 30.171159 # adx_mean
# data_mean[dmp_index] = 32.843816 # dmp_mean
# data_mean[dmn_index] = 32.276572 # dmn_mean
# data_mean[dayofweek_index] = 2   # day_of_week mean
# data_mean[edt_scaled_index] = 0.5 # edt_scaled
# data_mean[is_core_time_index] = 0.5 # is_core_time

# # how should mean for adx,dmp,and dmn be set?
# # print(data_mean)
# # data_std[0:4] = data_std[6:13] = 1 #data_std[3] # use close std for these columns
# data_std[close_batch_mean_list] = 1
# data_std[cdl_list] = 1
# data_std[zero_mean_one_var_list] = 1
# # data_std[volume_index] = 0
# # data_std[trade_count_index] = 0 # how should volume and trade_count be normalized?
# data_std[rsi_index] = 10 # rsi_std
# data_std[cci_index] = 100 # cci_std
# data_std[adx_index] = 16.460923 # adx_std
# data_std[dmp_index] = 18.971341 # dmp_std
# data_std[dmn_index] = 18.399032 # dmn_std
# data_std[dayofweek_index] = 1.414 # day_of_week std
# data_std[edt_scaled_index] = 1.25 # edt_scaled
# data_std[is_core_time_index] = 1 # is_core_time
# # # how should std for adx,dmp,and dmn be set?

# # As feature num increase, it is becoming tedious to maintain mean&variance for special feature. Will need new structure for updateing this in future.

# # print("data_mean shape: ", data_mean.shape)
# data_norm = (data - data_mean) / data_std