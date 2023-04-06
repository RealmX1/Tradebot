import pandas as pd
import numpy as np

start_sim_time = 1679612340 # 2023-03-23

# read the CSV file
df = pd.read_csv("nvda_1min_complex.csv")

# drop the last 4 columns
# df = df.iloc[:, :-4]

# df = df[df[:,0].argsort()] # sort by time; this is probably not necessary, but I included it just in case.
# print(df.shape)
# idx = (df['time'] > 1679612340).idxmax() # find the index of the first row with time > 1679612340 (whcih is the last training time)
# print(idx)
# df = df.loc[idx+1:]

df = df.iloc[42:, [i for i in range(df.shape[1]) if i not in [18, 19]]]

# save the modified file
df.to_csv("nvda_1min_complex_fixed.csv", index=False)