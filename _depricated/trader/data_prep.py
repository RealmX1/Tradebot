from data_utils import *


max_window_size = 32 # using how much data from the past to make prediction?
data_prep_window = max_window_size + 1 # becuase we need to remove 
close_idx = 3 # after removing time column
train_percent = 0.8
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


df = pd.read_csv('nvda_1min_sim.csv')
df_float = df.values.astype(float)
df_float = df_float[:,1:] # remove the utftime column.

print(df_float.shape)
feature_means = np.mean(df_float, axis=0,keepdims = True) # shape (1,6)
close_mean = feature_means[0,close_idx]
feature_stds = np.std(df_float, axis=0,keepdims = True) # shape (1,6)
close_stds = feature_stds[0,close_idx]
# print("means: ",feature_means)
# print("stds: ",feature_stds)

# normalization
# df_float = (df_float - feature_means) / feature_stds
# print(df_float.shape)

data = result = sample_z_continuous(df_float, data_prep_window)
print(data.shape)

# train_size = int(len(data) * train_percent)
# test_size = int(len(data) * (1-train_percent))
# # note that since we are predicting future stock data, learn on nearer data, and test on a previous data.
# train_data = data[test_size:,:,:]
# test_data = data[:test_size,:,:]

# train_dataset = NvidiaStockDataset(train_data)
# test_dataset = NvidiaStockDataset(test_data)
# total_dataset = NvidiaStockDataset(data)

# print("loading data")
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False) 
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
# total_loader = DataLoader(total_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

def prediction2price(prediction):
    return prediction * close_stds + close_mean

def get_next_stock_price(time, identifier = ""):
    # TODO: match identifier
    future_price = torch.tensor(data[time:time+1,-1,0]).to(device)
    return future_price

# returns a torch tensor of shape (1,feature_num)
def get_stock_price(time, identifier = ""):
    future_price = torch.tensor(data[time:time+1,-2,0]).to(device)
    return future_price

# returns a torch tensor of shape (1,window,feature_num)
def get_price_hist(time, identifier = "", window_size = 1):
    # TODO: should the window param be kept?
    if window_size > max_window_size:
        raise ValueError(f"window size is too large (larger than {max_window_size})")
        return None
    else:
        inputs = torch.tensor(data[time:time+1,max_window_size-window_size:-1,:]).to(device)
        print(inputs.shape)
        return inputs

def main():
    print("Test")
    print(get_next_stock_price(0))
    print(get_next_stock_price(2))
    print(get_stock_price(0))
    print(get_stock_price(2))
    print(get_price_hist(0))
    print(get_price_hist(2))

if __name__ == '__main__':
    main()