# attempt to predict up or down in 1~10min
# for now it performs than the one which predicts the price in 1~10min; despit that one isn't trained to predict up & down movement of stock.

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
model_path = cwd+"/lstm_updown.pt"

# Load the CSV file into a Pandas dataframe
# time,open,high,low,close,Volume,Volume MA

close_idx = 3 # after removing time column

df = pd.read_csv('nvda_1min_complex_fixed.csv')

# Define hyperparameters
input_size = 32 # Number of features (i.e. columns) in the CSV file -- the time feature is removed.
hidden_size = 32 # Number of neurons in the hidden layer of the LSTM

num_layers = 2 # Number of layers in the LSTM
output_size = 10 # Number of output values (closing price 1~10min from now)
learning_rate = 0.001
num_epochs = 10000
batch_size = 2048

window_size = 50 # using how much data from the past to make prediction?
data_prep_window = window_size + 10 # +10 becuase we need to keep 10 for calculating loss
drop_out = 0.1

train_percent = 0.8

# Define the LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size, output_size)
        self.fc2 = nn.Softmax(2)

    def forward(self, x): # assumes that x is of shape (batch_size,time_steps, features) 
        batch_size = x.shape[0]
        out, _ = self.lstm(x.float())
        out = self.fc1(out)
        predictions = self.fc2(out)
        # print("predictions shape: ",predictions.shape)
        # [2048, 32, 10]
        return predictions[:,-1,:]
        

# Define the dataset class
class NvidiaStockDataset(Dataset):
    def __init__(self, data):
        self.x = data[:,:-10,:] # slicing off the last entry of input
        self.y = data[:,window_size:,close_idx] # moving the target entry one block forward
        self.y = self.y - self.x[:,-1,close_idx:close_idx+1]
        # turn negative to 0, positive to 1
        self.y = np.clip(self.y,0,np.inf) 
        self.y[self.y != 0] = 1

        print(self.x.shape)
        print(self.y.shape)
        # (16713, 32, 32)
        # (16713, 10)
        assert self.x.shape[0] == self.y.shape[0]

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.x[idx,:,:], self.y[idx,:]


# Prepare the data
def sample_z_continuous(arr, z):
    n = arr.shape[0] - z + 1
    result = np.zeros((n, z, arr.shape[1]))
    for i in range(n):
        result[i] = arr[i:i+z]
    return result




def main():
    # torch.backends.cudnn.benchmark = True # according to https://www.youtube.com/watch?v=9mS1fIYj1So, this speeds up cnn.

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
        
    loss_fn = nn.BCEWithLogitsLoss()
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
            for i, (inputs, targets) in enumerate(train_loader):
                avg_loss *= 0.95
                # print(inputs.shape)
                # print(targets.shape)
                # inputs   torch.Size([2048, 32, 6])
                # targets  torch.Size([2048, 32, 10])
                inputs = inputs.float().to(device)
                targets = targets.float().to(device)
                outputs = model(inputs) # pass batch size to LSTM    
                # print(outputs.shape)
                # outputs  torch.Size([2048, 10])        
                loss = loss_fn(outputs, targets) 
                if avg_loss == 0: 
                    avg_loss = loss 
                else: 
                    avg_loss = avg_loss + loss.item()*0.05
                optimizer.zero_grad() # when removed, it doesn't improve training speed when not using zero_grad
                loss.backward()
                optimizer.step()

                true_direction = targets.cpu()
                # print(true_direction)
                pred_direction = outputs.clone().detach().cpu()
                # print("Pred_pre: ",pred_direction)
                pred_direction = torch.round(pred_direction)
                # print("Pred: ",pred_direction)

                total_cells = true_direction.numpy().size
                differing_cells = np.count_nonzero(true_direction != pred_direction)
                # print(differing_cells)
                total_predictions += total_cells
                true_prediction += (total_cells - differing_cells)

            # percent_diff = differing_cells / total_cells * 100

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
            for i, (inputs, targets) in enumerate(test_loader):
                inputs = inputs.float().to(device)
                targets = targets.float().to(device)
                outputs = model(inputs) # pass batch size to LSTM
                loss = loss_fn(outputs, targets)

                true_direction = targets.cpu()
                # print(true_direction)
                pred_direction = outputs.clone().detach().cpu()
                pred_direction = np.clip(pred_direction,0.5,np.inf)
                pred_direction[pred_direction != 0] = 1
                # print("Pred: ",pred_direction)

                total_cells = true_direction.numpy().size
                differing_cells = np.count_nonzero(true_direction != pred_direction)
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
            
            # total_predictions = 0
            # true_prediction = 0
            # for i, (inputs, targets) in enumerate(total_loader):
                
            #     inputs = inputs.float().to(device)
            #     targets = targets.float().to(device)
            #     outputs = model(inputs) # pass batch size to LSTM
            #     if predictions is None:
            #         predictions = outputs[:,[0,4,9]]
            #     else:
            #         predictions = torch.cat((predictions, outputs[:,[0,4,9]]), dim=0)


            #     true_direction = targets - inputs[:,-1,close_idx:close_idx+1]
            #     true_direction = np.clip(true_direction.cpu(),0,1) # this turns negative to 0, positive to 1
            #     pred_direction = outputs - inputs[:,-1,close_idx:close_idx+1]
            #     pred_direction = np.clip(pred_direction.cpu(),0,1)

            #     total_cells = true_direction.numpy().size
            #     differing_cells = np.count_nonzero(true_direction != pred_direction)
            #     total_predictions += total_cells
            #     true_prediction += (total_cells - differing_cells)

            # test_loss /= len(test_loader)
            # predictions = predictions.cpu().numpy()
            # print(f'True prediction: {true_prediction/total_predictions*100:.2f}%')
            # print(f'Prediction completed in {time.time()-start_time:.2f} seconds')

                
            # # Plot the results
            # predictions = predictions * close_stds + close_mean
            # print(data.shape[0])
            # print(len(predictions))
            # plt.plot(df_float[window_size:,close_idx] * close_stds + close_mean, label='Actual')
            # plt.plot(predictions[:,0], label='1min')
            # plt.plot(predictions[:,1], label='5min')
            # plt.plot(predictions[:,2], label='10min')
            # plt.legend()
            # plt.axvline(x=test_size, color='r')
            # plt.show()
                    

        torch.save(model.state_dict(), model_path)
        print('Normal exit. Model saved.')
    except KeyboardInterrupt or Exception:
        # save the model if the training was interrupted by keyboard input
        torch.save(model.state_dict(), model_path)
        print('Model saved.')



if __name__ == '__main__':
    main()
    # cProfile.run('main()') # this shows execution time of each function. Might be useful for debugging & accelerating in detail.
