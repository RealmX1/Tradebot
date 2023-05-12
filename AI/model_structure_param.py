import torch
from datetime import datetime
# close_idx           = 3

feature_num         = input_size = 16  # candel  # Number of features (i.e. columns) in the CSV file -- the time feature is removed.
hidden_size         = 100    # Number of neurons in the hidden layer of the LSTM
num_layers          = 1   # Number of layers in the LSTM
output_size         = 1     # Number of output values (closing price 1~10min from now)
prediction_window   = 10
hist_window         = 60 # using how much data from the past to make prediction?
data_prep_window    = hist_window + prediction_window # +ouput_size becuase we need to keep 10 for calculating loss
dropout             = 0.1

weight_decay = 1
# weights = 

num_epochs          = 100
train_ratio = 0.9

learning_rate       = 0.00005
batch_size          = 10000

scheduler_patience = 20
scheduler_factor = 0.97
scheduler_threshold = 0.0000001

decision_hidden_size    = 100
action_num              = 4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model_type = 'lstm'

attention = True

if model_type == 'lstm':
    if attention:
        config_name = f'lstm_attention_{feature_num}f{num_layers}l{hidden_size}h{hist_window}hist{prediction_window}pred{weight_decay}decay'
    else:
        config_name = f'lstm_no_attention_{feature_num}f{num_layers}l{hidden_size}h{hist_window}hist{prediction_window}pred{weight_decay}decay'
elif model_type == 'transformer':
    config_name = f'transformer_{feature_num}f{num_layers}l{hidden_size}h{hist_window}hist{prediction_window}pred'

initial_capital = 100000

pct_pred_multiplier = 10 # applies to y data; also should affect threshold directly # note that by doing this y is 1/100, not 1/100

policy_threshold = 0.005 * pct_pred_multiplier


training_start = datetime(2020, 1, 1)
trianing_end = datetime(2023, 1, 1)
training_start_str = training_start.strftime('%Y-%m-%d')
training_end_str = trianing_end.strftime('%Y-%m-%d')
training_time_str = f'{training_start_str}_{training_end_str}'

testing_start = datetime(2023, 1, 1)
testing_end = datetime(2023, 5, 1)
testing_start_str = testing_start.strftime('%Y-%m-%d')
testing_end_str = testing_end.strftime('%Y-%m-%d')
testing_time_str = f'{testing_start_str}_{testing_end_str}'



training_symbols = ['AAPL','DQ', 'PDD', 'VZ']