from enum import Enum
class NormParam(Enum):
    CloseBatch = (0,1) # (mean, std)
    NoNorm = (0,1)
    Volume = (0,1)
    TradeCount = (0,1)
    RSI = (50,10)
    CCI = (0,100)
    ADX = (30.171159,16.460923)
    DMP = (32.843816,18.971341)
    DMN = (32.276572,18.399032)
    DayOfWeek = (2,1)
    EDT = (0.5,0.288675)
    IsCoreTime = (0.5,0.5)
    # how should std for adx,dmp,and dmn be set?

    CDL = (0,1)
    























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