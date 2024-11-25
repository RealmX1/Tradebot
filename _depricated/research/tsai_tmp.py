from tsai.all import *
import numpy as np
# start_time = time.time()
# print("Loading data...")
# ts = get_forecasting_time_series("Sunspots").values
# X, y = SlidingWindow(60, horizon=3)(ts)
# print(f"Data loaded in: {time.time() - start_time}")

# print(type(X), type(y), type(ts))
# print(X.shape, y.shape, ts.shape)
# print(ts[0:5])
# print(len(ts))

def moving_average(data, window_size):
    weights = np.repeat(1.0, window_size) / window_size
    return np.convolve(data, weights, mode='valid')

data = [1,2,3,2,5,6,7,8,9,10]
print(moving_average(data, 3))