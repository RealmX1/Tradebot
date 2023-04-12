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

n = 5
data = np.random.rand(10, n)
print(data)