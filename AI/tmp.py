import numpy as np

# # Generate a random input array
# x = np.random.rand(2, 3, 4)

# # Calculate mean and std along the second axis
# mean = np.mean(x, axis=1)
# std = np.std(x, axis=1)
# n=1

# # Calculate normalized arrays using both methods
# # norm1 = (x - mean) / std
# # Traceback (most recent call last):
# #   File "tmp.py", line 11, in <module>
# #     norm1 = (x - mean) / std
# # ValueError: operands could not be broadcast together with shapes (10,5,3) (10,3) 
# mean[:,-n:] = 0
# std[:,-n:] = 1
# norm2 = (x - mean[:, None, :]) / std[:, None, :]

# # Check if the two methods produce the same result
# print(norm2)



# import pandas as pd
# import pytz

# pd.set_option('display.max_rows', None)

# # load data
# df = pd.read_csv('nvda_1min_complex_fixed.csv')

# # Convert the Unix timestamp column to datetime
# df['time'] = pd.to_datetime(df['time'], unit='s')

# # Convert the timezone to EDT
# df['time'] = df['time'].dt.tz_localize('UTC').dt.tz_convert('US/Eastern')

# df_float = df.values #.astype(float)
# df_float = df_float[:,1:18] # remove the utftime column, and columns after rsi-based-ma.
# global_stadardization = df_float[:,-4]
# mean = np.zeros(4)
# mean[0] = np.mean(global_stadardization, axis = 0)
# mean[1] = mean[0]
# mean[2] = mean[3] = 50 # rsi fluctuates between 0 and 100
# std = np.ones(4)
# std[0] = np.std(global_stadardization, axis = 0)
# std[1] = std[0]
# std[2] = std[3] = 10 # rsi fluctuates between 0 and 100


# # Print the updated dataframe
# print(mean)
# print(std)



# import torch
# import time

# # Create some dummy data
# target = torch.randn(100, 10, 5)

# # Test approach 1: using slicing and indexing
# start_time = time.time()
# x1 = target[:, 0:1, :]
# end_time = time.time()
# print("Approach 1: %.8f seconds" % (end_time - start_time))

# # Test approach 2: using unsqueeze
# start_time = time.time()
# x2 = target[:, 0, :].unsqueeze(1)
# end_time = time.time()
# print("Approach 2: %.8f seconds" % (end_time - start_time))

# '''
# TEST RESULT: SLICING IS FASTER
# Approach 1: 0.00180817 seconds
# Approach 2: 0.00305629 seconds
# '''

# n = 5
# data = np.random.rand(10, n)
# print(data)
# import json
# with open('training_param_log.json', 'w') as f:
#         json.dump({'learning_rate': 0.0000119878, 'best_prediction': 67.53}, f)

# import matplotlib.pyplot as plt

# values = [0, 1, 0, 1, 0, 1, 0, 0, 1]

# # Filter non-zero values

# # Create x-axis positions for non-zero values
# positions = [i for i, x in enumerate(values) if x != 0]
# values = [x for x in values if x != 0]

# # Plot non-zero values at their respective x-positions
# plt.plot(positions, values, 'ro')

# plt.show()



'''
    test whether global noramlization will change target Y -- Y need to represent percentage change of the original X.
'''
# import random
# np.random.seed(42)
# batch_size = 100



# data = np.random.rand(batch_size*10, 5, 4)
# data_mean = np.mean(data, axis = 0)
# data_std = np.std(data, axis = 0)
# data_norm = (data-data_mean)/data_std

# batch_data = data_norm[:batch_size,:,:]

# close_idx = 2
# prediction_window = 2
# x_raw = batch_data[:,:-prediction_window,:]
# y_raw = batch_data[:,-prediction_window:,close_idx] 
# tmp = x_raw[:,-1,close_idx:close_idx+1]
# print(tmp.shape)

# y = (y_raw - tmp)/tmp * 100
# print(y.shape)

# print(y[:5,:])

# import gc
# import torch
# torch.cuda.empty_cache()
# gc.collect()


# importing libraries
# import matplotlib.pyplot as plt
  
# # creating data
# xdata = [0, 2, 4, 6, 8, 10, 12, 14]
# ydata = [4, 2, 8, 6, 10, 5, 12, 6]
  
# # plotting data
# plt.plot(xdata, ydata, ls=None)
  
# # Displaying plot
# plt.show()

# import torch
# discount_rate = 0.80
# prediction_window = 10
# discount_factors = torch.pow(torch.tensor(discount_rate), torch.arange(prediction_window).float()) #.to(device)
# print(discount_factors)

# import pandas as pd 
# pd.set_option('display.max_columns', None)
# np.set_printoptions(precision=3, suppress=True)


# data_path = "data/aapl_test.csv"
# df = pd.read_csv(data_path, index_col = ['symbol', 'timestamp'])

# close_batch_mean_list = []
# BB_mask = df.columns.str.contains("BBL|BBM|BBU").tolist() # bolinger band
# MA_mask = df.columns.str.contains("MA").tolist() # moving average
# CDL_mask = df.columns.str.contains("open|high|low|close|vwap").tolist()
# close_batch_mean_list = [a or b or c for a, b, c in zip(BB_mask, MA_mask, CDL_mask)]
# # print(close_batch_mean_list)
# close_batch_mean_list = np.where(close_batch_mean_list)[0] # index 0 is needed to unpack return from np.where, which is a tuple of lists.
# print(close_batch_mean_list)

# data = df.values
# data_mean = np.mean(data, axis = 0)
# data_std = np.std(data, axis = 0)
# # print(data_mean.shape)

# volume_mask = df.columns.str.contains("volume").tolist()
# volume_index = np.where(volume_mask)[0][0]
# trade_count_mask = df.columns.str.contains("trade_count").tolist()
# trade_count_index = np.where(trade_count_mask)[0][0]
# rsi_mask = df.columns.str.contains("RSI").tolist()
# rsi_index = np.where(rsi_mask)[0][0]
# cci_mask = df.columns.str.contains("CCI").tolist()
# cci_index = np.where(cci_mask)[0][0]
# adx_mask = df.columns.str.contains("ADX").tolist()
# adx_index = np.where(adx_mask)[0][0]
# dmp_mask = df.columns.str.contains("DMP").tolist()
# dmp_index = np.where(dmp_mask)[0][0]
# dmn_mask = df.columns.str.contains("DMN").tolist()
# dmn_index = np.where(dmn_mask)[0][0]
# dayofweek_mask = df.columns.str.contains("dayofweek").tolist()
# dayofweek_index = np.where(dayofweek_mask)[0][0]
# edt_scaled_mask = df.columns.str.contains("edt_scaled").tolist()
# edt_scaled_index = np.where(edt_scaled_mask)[0][0]
# is_core_time_mask = df.columns.str.contains("is_core_time").tolist()
# is_core_time_index = np.where(is_core_time_mask)[0][0]


# # data_mean[0:4] = data_mean[6:13] = 0 # data_mean[3] # use close mean for these columns
# data_mean[close_batch_mean_list] = 0
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
# print("normalized data: \n", data_norm[:5,:])

import pandas_ta as ta
import pandas as pd

# def locate_cols(strings_list, substring):
#     return [i for i, string in enumerate(strings_list) if substring in string]

# df = pd.read_csv('data/baba_test_test.csv', index_col = ['symbol', 'timestamp'])
# df.ta.macd(append = True)
# df = df.dropna()

# col_locs = locate_cols(df.columns.tolist(), 'MACD')
# print(col_locs)

# df.to_csv('data/baba_test_test_macd.csv', index=True, index_label=['symbol', 'timestamp'])
# print(df.columns.tolist())
# # print(df.head(20))

# a = np.array([True, False, True])
# b = np.array([False, True, True])

# c = a & b

# print(c)

data_path = '../data/csv/last_week_bar_set_20230410_20230417_raw.csv'
df = pd.read_csv(data_path, index_col = ['symbol', 'timestamp'])    
print(df.head(5))



# testing capbility of github-copiolet
# this is a piece of code that calculates the number of miss match between two lists
a = np.array([1,2,3,4,5,6,7,8,9,10])
b = np.array([1,2,3,4,5,6,7,8,9,10])

c = a == b
