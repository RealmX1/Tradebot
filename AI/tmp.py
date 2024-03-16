import numpy as np

pred_window = 10
batch_size = 5

a = np.full((pred_window,), batch_size)
print(a)