import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch # PyTorch
import torch.nn as nn # PyTorch neural network module
from torch.utils.data import Dataset, DataLoader # PyTorch data utilities
from torch.optim.lr_scheduler import StepLR # PyTorch learning rate scheduler
from torch.optim import AdamW
# from apex.optimizers import FusedLAMB

import matplotlib.pyplot as plt
import os
import numpy as np
import time
import cProfile

cwd = os.getcwd()
model_path = cwd+"/lstm.pt"

# Load the CSV file into a Pandas dataframe
# time,open,high,low,close,Volume,Volume MA

close_idx = 3 # after removing time column

df = pd.read_csv('nvda_1min_complex_fixed.csv')

# Define hyperparameters
feature_num = input_size = 32 # Number of features (i.e. columns) in the CSV file -- the time feature is removed.
hidden_size = 64 # Number of neurons in the hidden layer of the LSTM

num_layers = 8 # Number of layers in the LSTM
output_size = 10 # Number of output values (closing price 1~10min from now)
learning_rate = 0.0001
num_epochs = 2000
batch_size = 2048

window_size = 50 # using how much data from the past to make prediction?
data_prep_window = window_size + output_size # +ouput_size becuase we need to keep 10 for calculating loss
drop_out = 0.1

train_percent = 0.8

no_norm_num = 4 # the last four column of the data are 0s and 1s, no need to normalize them (and normalization might cause 0 division problem)

# Define the LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size, output_size)

    def forward(self, x): # assumes that x is of shape (batch_size,time_steps, features) 
        batch_size = x.shape[0]
        tmp, _ = self.lstm(x) #.float()
        output = self.fc1(tmp)
        # print(output.shape)
        # output.shape
        return output[:,-1,:]
        

# Define the dataset class
# data.shape: (data_num, data_prep_window, feature_num)
class NvidiaStockDataset(Dataset):
    def __init__(self, data):
        self.x = data[:,:-output_size,:] # slicing off the last entry of input
        # print("x.shape: ",self.x.shape)
        # x.shape: (data_num, window_size, feature_num)
        self.y = data[:,window_size:,close_idx] # moving the target entry one block forward
        # y.shape: (data_num, output_size)
        self.x_mean = np.mean(self.x, axis=1)
        self.x_std = np.std(self.x, axis=1)
        self.x_mean[:,-no_norm_num:] = 0
        self.x_std[:,-no_norm_num:] = 1
        # print("x_mean.shape: ", self.x_mean.shape)
        # mean/std.shape: (data_num, feature_num)

        # does this normalization broadcast work properly? 
        # desired effect is x[i,:,j] will be normalized using x_mean[i,j] and x_std[i,j],
        # and y[i,j] will be normalized using x_mean[i,close_idx] and x_std[i,close_idx]
        self.x = (self.x - self.x_mean[:,None,:]) / self.x_std[:,None,:]
        self.y = (self.y - self.x_mean[:,close_idx:close_idx+1]) / self.x_std[:,close_idx:close_idx+1]

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.x[idx,:,:], self.y[idx,:], self.x_mean[idx,:], self.x_std[idx,:]


# Prepare the data
def sample_z_continuous(arr, z):
    n = arr.shape[0] - z + 1
    result = np.zeros((n, z, arr.shape[1]))
    for i in range(n):
        result[i] = arr[i:i+z]
    return result


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


def main():
    # torch.backends.cudnn.benchmark = True # according to https://www.youtube.com/watch?v=9mS1fIYj1So, this speeds up cnn.

    df_float = df.values #.astype(float)
    df_float = df_float[:,1:] # remove the utftime column.
    # data_num = df_float.shape[0]
    print("df_float shape:",df_float.shape)
    # (data_num, output_size)

    # # old normalization -- normalize the entire dataset at once
    # feature_means = np.mean(df_float, axis=0,keepdims = True) # shape (1,6)
    # close_mean = feature_means[0,close_idx]
    # feature_stds = np.std(df_float, axis=0,keepdims = True) # shape (1,6)
    # close_stds = feature_stds[0,close_idx]
    # # print("means: ",feature_means)
    # # print("stds: ",feature_stds)
    #df_float = (df_float - feature_means) / feature_stds
    # NOW THE NORMALIZATION IS DONE IN THE DATASET CLASS, at each prediction input/history window
    
    # am I doing this correctly? Should LSTM be trained this way?
    # or should it be trained using continuous dataset, and progress by feeding one data after another?
    # at least current method makes it easier to noramlize each window of input independently.
    data = result = sample_z_continuous(df_float, data_prep_window)
    print(data.shape)
    # (data_num, data_prep_window, output_size)

    train_size = int(len(data) * train_percent)
    test_size = int(len(data) * (1-train_percent))
    # learn on nearer data, and test on a previous data; 
    # not sure which order is better... don't have knowledge of such metric; probably should do experiment and read paper on this
    # train_data = data[:train_size,:,:]
    # test_data = data[train_size:,:,:]
    train_data = data[test_size:,:,:]
    test_data = data[:test_size,:,:]

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


    print("loading model")
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTM(input_size, hidden_size, num_layers, output_size, drop_out).to(device)
    if os.path.exists(model_path):
        print("Loading existing model")
        model.load_state_dict(torch.load(model_path))
    else:
        print("No existing model")
    print(model)
    print(f'model loading completed in {time.time()-start_time:.2f} seconds')
        
    loss_fn = nn.MSELoss()
    optimizer = AdamW(model.parameters(),weight_decay=0.01, lr=learning_rate)
    # Define the StepLR scheduler to decrease the learning rate by a factor of gamma every step_size epochs
    scheduler = StepLR(optimizer, step_size=num_epochs/50, gamma=0.95)


    try:
        # Train the model
        print("Training model")
        model.train()
        start_time = time.time()
        total_predictions = 0
        true_prediction = 0
        avg_loss = 0
        for epoch in range(num_epochs):
            total_predictions *= 0.9
            true_prediction *= 0.9
            for i, (inputs, targets, _, _) in enumerate(train_loader):
                avg_loss *= 0.95
                # print(inputs.shape)
                # print(targets.shape)
                # inputs   torch.Size([2048, 32, 6])
                # targets  torch.Size([2048, 32, 1])
                inputs = inputs.float().to(device) # probably need to do numpy to pt tensor in the dataset; need testing on efficiency #!!! CAN"T BE DONE. before dataloarder they are numpy array, not torch tensor
                targets = targets.float().to(device)
                outputs = model(inputs) # pass batch size to LSTM    
                # print(outputs.shape)
                # outputs  torch.Size([2048, 1])        
                loss = loss_fn(outputs, targets) 
                
                optimizer.zero_grad() # when removed, it doesn't improve training speed when not using zero_grad
                loss.backward()
                optimizer.step()

                total_cells, differing_cells = get_direction_diff(inputs, targets, outputs)
                total_predictions += total_cells
                true_prediction += (total_cells - differing_cells)
                
                avg_loss = avg_loss + loss.item()*0.05

            percent_diff = differing_cells / total_cells * 100

            # scheduler.step()
            print(f'Epoch {epoch+1:3}/{num_epochs:3}, Training Loss: {avg_loss:10.7f}, Time per epoch: {(time.time()-start_time)/(epoch+1):.2f} seconds, Correct Prediction: {true_prediction/total_predictions*100:.2f}%') # Learning Rate: {scheduler.get_last_lr()[0]:9.6f}
        print(f'Training completed in {time.time()-start_time:.2f} seconds')

        # Test the model
        model.eval() 
        test_loss = 0
        start_time = time.time()
        total_predictions = 0
        true_prediction = 0
        with torch.no_grad():
            i = 0
            for i, (inputs, targets, _, _) in enumerate(test_loader):
                inputs = inputs.float().to(device)
                targets = targets.float().to(device)
                outputs = model(inputs) # pass batch size to LSTM
                loss = loss_fn(outputs, targets)

                total_cells, differing_cells = get_direction_diff(inputs, targets, outputs)
                total_predictions += total_cells
                true_prediction += (total_cells - differing_cells)

                # percent_diff = differing_cells / total_cells * 100


                test_loss += loss.item()
                if i % 1000 == 0:
                    print(f"Testing Loss: {loss.item():10.4f}")
            print("i is ",i)
            test_loss /= len(test_loader)
            print(f'Test Loss: {test_loss:.4f}')
            print(f'True prediction: {true_prediction/total_predictions*100:.2f}%')
        print(f'Testing completed in {time.time()-start_time:.2f} seconds')
            

        # Make predictions
        model.eval() 
        start_time = time.time()

        predictions = None
        with torch.no_grad():
            correct=[0]*3
            total=[0]*3
            
            total_predictions = 0
            true_prediction = 0
            for i, (inputs, targets, mean, std) in enumerate(total_loader):
                
                inputs = inputs.float().to(device)
                targets = targets.float().to(device)
                mean = mean.float().to(device)
                std = std.float().to(device)
                outputs = model(inputs) # pass batch size to LSTM

                # print("outputs: ",outputs[:,[0,4,9]].shape)
                # print("mean: ",mean[:,close_idx].shape)
                prediction = outputs[:,[0,4,9]] * std[:,close_idx:close_idx+1] + mean[:,close_idx:close_idx+1]
                if predictions is None:
                    predictions = prediction
                else:
                    predictions = torch.cat((predictions, prediction), dim=0)


                total_cells, differing_cells = get_direction_diff(inputs, targets, outputs)
                total_predictions += total_cells
                true_prediction += (total_cells - differing_cells)

            test_loss /= len(test_loader)
            predictions = predictions.cpu().numpy()
            print(f'True prediction: {true_prediction/total_predictions*100:.2f}%')
            print(f'Prediction completed in {time.time()-start_time:.2f} seconds')

                
            # Plot the results
            assert data.shape[0] == len(predictions)
            print(data.shape[0])
            print (predictions)
            plt.plot(df_float[window_size:,close_idx], label='Actual')
            x = np.arange(len(predictions))
            plt.plot(x+1, predictions[:,0], label='1min')
            plt.plot(x+5, predictions[:,1], label='5min')
            plt.plot(x+10, predictions[:,2], label='10min')
            plt.legend()
            plt.axvline(x=test_size, color='r')
            plt.show()
                    

        torch.save(model.state_dict(), model_path)
        print('Normal exit. Model saved.')
    except KeyboardInterrupt or Exception:
        # save the model if the training was interrupted by keyboard input
        torch.save(model.state_dict(), model_path)
        print('Model saved.')



if __name__ == '__main__':
    main()
    # cProfile.run('main()') # this shows execution time of each function. Might be useful for debugging & accelerating in detail.
