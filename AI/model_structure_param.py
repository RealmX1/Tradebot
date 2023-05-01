close_idx           = 3

feature_num         = input_size = 23  # candel  # Number of features (i.e. columns) in the CSV file -- the time feature is removed.
hidden_size         = 100    # Number of neurons in the hidden layer of the LSTM
num_layers          = 1    # Number of layers in the LSTM
output_size         = 1     # Number of output values (closing price 1~10min from now)
prediction_window   = 5
hist_window         = 30 # using how much data from the past to make prediction?
data_prep_window    = hist_window + prediction_window # +ouput_size becuase we need to keep 10 for calculating loss
dropout             = 0.1


learning_rate       = 0.00005
batch_size          = 10000