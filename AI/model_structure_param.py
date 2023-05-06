import torch
# close_idx           = 3

feature_num         = input_size = 16  # candel  # Number of features (i.e. columns) in the CSV file -- the time feature is removed.
hidden_size         = 200    # Number of neurons in the hidden layer of the LSTM
num_layers          = 1   # Number of layers in the LSTM
output_size         = 1     # Number of output values (closing price 1~10min from now)
prediction_window   = 5
hist_window         = 30 # using how much data from the past to make prediction?
data_prep_window    = hist_window + prediction_window # +ouput_size becuase we need to keep 10 for calculating loss
dropout             = 0.1

weight_decay = 3
# weights = 

learning_rate       = 0.00005
batch_size          = 20000

scheduler_patience = 15
scheduler_factor = 0.96
scheduler_threshold = 0.0000005

decision_hidden_size    = 100
action_num              = 4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

attention = True
if attention:
    config_name = f'lstm_{"attention_"}{feature_num}f{num_layers}l{hidden_size}h{hist_window}hist{prediction_window}pred{weight_decay}decay'

initial_capital = 100000

pct_pred_multiplier = 10 # applies to y data; also should affect threshold directly

policy_threshold = 0.005 * pct_pred_multiplier

training_data_path = f'../data/csv/bar_set_20200101_20230504_AAPL_{feature_num}feature0.csv'

