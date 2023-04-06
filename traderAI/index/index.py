import warnings
import numpy as np


# calculate moving average along the first axis (assumes the first axis is time)
def moving_average (close_price, axis = 0, window = 10):
    history_len = len(close_price)
    
    if window > len:
        # Print a warning message
        warnings.warn("window is longer than data length; changed to data length")

    return np.mean(close_price[-window:],axis = 0)

# random_numbers = np.random.randn(20,2)
# print(rnp.mean(random_numbers[-10:,:]))

# create a 2D numpy array with 3 rows and 4 columns
arr = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]])
print(arr.shape)

# calculate mean of all elements
mean_all = np.mean(arr)
print("Mean of all elements:", mean_all)

# calculate mean along axis 0 (column-wise)
mean_axis0 = np.mean(arr[], axis=0)
print("Mean along axis 0:", mean_axis0)

# calculate mean along axis 1 (row-wise)
mean_axis1 = np.mean(arr, axis=1)
print("Mean along axis 1:", mean_axis1)