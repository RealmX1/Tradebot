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

import torch
discount_rate = 0.80
prediction_window = 10
discount_factors = torch.pow(torch.tensor(discount_rate), torch.arange(prediction_window).float()) #.to(device)
print(discount_factors)