import torch.nn as nn
from AI.lstm_updown_old import *
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from torch.utils.data import Dataset, DataLoader # PyTorch data utilities
from torch.optim.lr_scheduler import StepLR # PyTorch learning rate scheduler
from torch.optim import AdamW

# model = LSTM(input_size, hidden_size, num_layers, output_size, drop_out)
# if os.path.exists(model_path):
#     print("Loading existing model")
#     model.load_state_dict(torch.load(model_path))
# else:
#     print("No existing model")

# total_params = sum(p.numel() for p in model.parameters())
# print(f"Total number of parameters: {total_params}")



# compare pytorch dataset,dataloader with tensorflow dataset,iterator

close_idx = 3 # after removing time column

df = pd.read_csv('nvda_1min_complex_fixed.csv')

# Define hyperparameters
input_size = 32 # Number of features (i.e. columns) in the CSV file -- the time feature is removed.
hidden_size = 64 # Number of neurons in the hidden layer of the LSTM

num_layers = 8 # Number of layers in the LSTM
output_size = 10 # Number of output values (closing price 1~10min from now)
learning_rate = 0.0001
num_epochs = 0
batch_size = 2048

window_size = 32 # using how much data from the past to make prediction?
data_prep_window = window_size + 10 # +10 becuase we need to keep 10 for calculating loss
drop_out = 0.1

train_percent = 0.8

# Prepare the data
def sample_z_continuous(arr, z):
    n = arr.shape[0] - z + 1
    result = np.zeros((n, z, arr.shape[1]))
    for i in range(n):
        result[i] = arr[i:i+z]
    return result

df_float = df.values.astype(float)
df_float = df_float[:,1:] # remove the utftime column.
print(df_float.shape)
feature_means = np.mean(df_float, axis=0,keepdims = True) # shape (1,6)
close_mean = feature_means[0,close_idx]
feature_stds = np.std(df_float, axis=0,keepdims = True) # shape (1,6)
close_stds = feature_stds[0,close_idx]
# print("means: ",feature_means)
# print("stds: ",feature_stds)

df_float = (df_float - feature_means) / feature_stds
print(df_float.shape)
# (21513, 6)
data = result = sample_z_continuous(df_float, data_prep_window)
print(data.shape)
# (21481, 33, 6)

train_size = int(len(data) * train_percent)
test_size = int(len(data) * (1-train_percent))
# note that since we are predicting future stock data, learn on nearer data, and test on a previous data.
# train_data = data[:train_size,:,:]
# test_data = data[train_size:,:,:]
train_data = data[test_size:,:,:]
test_data = data[:test_size,:,:]



########### tensorflow dataset ###########

import tensorflow as tf
print(tf.__version__)
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Load data
x_train = train_data[:,:-10,:] # slicing off the target data
y_train = train_data[:,window_size:,close_idx] # moving the target entry one block forward

# Create tf.data.Dataset
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

# Apply transformations
dataset = dataset.shuffle(buffer_size=len(x_train))
dataset = dataset.batch(batch_size)
dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# Iterate over the dataset
for x_batch, y_batch in dataset:
    print("tf inputs: ", x_batch.shape)
    print("tf targets: ", y_batch.shape)
    print(x_batch.dtype)
    print(type(x_batch))
    break




########### torch dataset ###########

# Define the dataset class
class NvidiaStockDataset(Dataset):
    def __init__(self, data):
        self.x = data[:,:-10,:] # slicing off the last entry of input
        self.y = data[:,window_size:,close_idx] # moving the target entry one block forward
        # print(self.x.shape)
        # print(self.y.shape)
        # (16713, 32, 32)
        # (16713, 10)
        assert self.x.shape[0] == self.y.shape[0]

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.x[idx,:,:], self.y[idx,:]


def get_direction_diff(inputs,targets,outputs):
    true_direction = targets - inputs[:,-1,close_idx:close_idx+1]
    true_direction = np.clip(true_direction.cpu(),0,np.inf) # this turns negative to 0, positive to 1
    true_direction[true_direction != 0] = 1
    pred_direction = outputs - inputs[:,-1,close_idx:close_idx+1]
    pred_direction = np.clip(pred_direction.clone().detach().cpu(),0,np.inf)
    pred_direction[pred_direction != 0] = 1
    # print("True: ", true_direction)
    # print("Pred: ", pred_direction)

    total_cells = true_direction.numpy().size
    differing_cells = np.count_nonzero(true_direction != pred_direction)

    return total_cells, differing_cells



# torch.backends.cudnn.benchmark = True # according to https://www.youtube.com/watch?v=9mS1fIYj1So, this speeds up cnn.

train_dataset = NvidiaStockDataset(train_data)
test_dataset = NvidiaStockDataset(test_data)
total_dataset = NvidiaStockDataset(data)

print("loading data")
start_time = time.time()
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False) 
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
total_loader = DataLoader(total_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
# testing have shown that my gtx1080ti doesn't benefit from changing num_worker; but future hardware might need them.
print(f'data loading completed in {time.time()-start_time:.2f} seconds')


for i, (inputs, targets) in enumerate(train_loader):
    print("torch inputs: ", inputs.shape)
    print("torch targets: ", targets.shape)
    print(inputs.dtype)
    print(type(inputs))
    break
