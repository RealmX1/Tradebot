import pandas as pd
import numpy as np
import torch
import time
from data_utils import *
from model_structure_param import *

df = pd.read_csv('../data/csv/bar_set_huge_20200101_20230417_AAPL_23feature.csv', index_col = ['symbol', 'timestamp'])
print(df.shape)
# print(df.columns.contains("BBL_|BBM_|BBU_"))
tmp = df.columns.str
print(tmp.contains("BBL_|BBM_|BBU_"))
# data = df.values
# data = sample_z_continuous(data, data_prep_window)
# print(data.shape)