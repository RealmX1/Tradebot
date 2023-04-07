# UNFINISHED
# here I'm going to try build a regular transformer model for test genreation. Later, this will be adapted to stock prediction that outputs

import pandas as pd
import os
  
names = ['year', 'month', 'day', 'dec_year', 'sn_value' , 
         'sn_error', 'obs_num', 'extra']
df = pd.read_csv(
    "https://data.heatonresearch.com/data/t81-558/SN_d_tot_V2.0.csv",
    sep=';',header=None,names=names,
    na_values=['-1'], index_col=False)

# We want to find the starting index where the missing observation (obs_num == 0) no longer occurs.
start_id = max(df[df['obs_num'] == 0].index.tolist())+1  
# print(start_id)
df = df[start_id:] # Trim the rows that have missing observations

# Divide into training and test/validation sets.
df['sn_value'] = df['sn_value'].astype(float)
df_train = df[df['year']<2000]
df_test = df[df['year']>=2000]

spots_train = df_train['sn_value'].tolist()
spots_test = df_test['sn_value'].tolist()

# print("Training set has {} observations.".format(len(spots_train)))
# print("Test set has {} observations.".format(len(spots_test)))






import numpy as np

def to_sequences(seq_size, obs):
    x = []
    y = []

    for i in range(len(obs)-SEQUENCE_SIZE):
        #print(i)
        window = obs[i:(i+SEQUENCE_SIZE)]
        after_window = obs[i+SEQUENCE_SIZE]
        window = [[x] for x in window]
        #print("{} - {}".format(window,after_window))
        x.append(window)
        y.append(after_window)
        
    return np.array(x),np.array(y)
    
    
SEQUENCE_SIZE = 10
x_train,y_train = to_sequences(SEQUENCE_SIZE,spots_train)
x_test,y_test = to_sequences(SEQUENCE_SIZE,spots_test)

print("Shape of training set: {}".format(x_train.shape))
print("Shape of test set: {}".format(x_test.shape))

print(x_train[0])









# I probably should build a reference LSTM predictor and a pure Fully connected network.

# here we 