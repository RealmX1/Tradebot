import os

from AI.model_structure_param import *

time_str = '20200101_20200701'
symbol = 'MSFT'
data_type = '16feature0'
data_name = f'bar_set_{time_str}_{symbol}_{data_type}_RAW'
model_name = f'last_model_{config_name}'


log_pth_template = f'../TradebotGraph/{data_name}--{model_name}_{{}}.txt'
pic_pth_template = f'../TradebotGraph/{data_name}--{model_name}_{{i_th_attempt}}_{{block_col_name}}.png'

i = 0
while True:
    i += 1
    pic_pth_template_2 = pic_pth_template.format(i_th_attempt = i, block_col_name = '{}')

    if not os.path.exists(complete_log_pth := log_pth_template.format(i)): break
print(i)
print(complete_log_pth)
print(pic_pth_template_2)
print(pic_pth_template_2.format('N/A'))
