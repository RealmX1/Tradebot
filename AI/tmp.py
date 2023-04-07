import numpy as np

# Generate a random input array
x = np.random.rand(2, 3, 4)

# Calculate mean and std along the second axis
mean = np.mean(x, axis=1)
std = np.std(x, axis=1)
n=1

# Calculate normalized arrays using both methods
# norm1 = (x - mean) / std
# Traceback (most recent call last):
#   File "tmp.py", line 11, in <module>
#     norm1 = (x - mean) / std
# ValueError: operands could not be broadcast together with shapes (10,5,3) (10,3) 
mean[:,-n:] = 0
std[:,-n:] = 1
norm2 = (x - mean[:, None, :]) / std[:, None, :]

# Check if the two methods produce the same result
print(norm2)