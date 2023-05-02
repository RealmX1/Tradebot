import pickle
import re
import numpy as np
import pandas as pd

# with open('lists_no_multithread_0to12.pkl', 'rb') as f:
#     loaded_lists = pickle.load(f)

# block_str_lst, end_strs_lst, loss_lst = loaded_lists

with open('lists_no_multithread.pkl', 'rb') as f:
    loaded_lists = pickle.load(f)
block_str_lst, end_strs_lst, loss_lst = loaded_lists
# block_str_lst2, end_strs_lst2, loss_lst2 = loaded_lists
# block_str_lst += block_str_lst2
# end_strs_lst += end_strs_lst2
# loss_lst += loss_lst2

df = pd.read_csv('../data/feature_efficacy_test.csv')
print(len(block_str_lst))


end_str_1_pattern = r'decision:[\s]*(?P<decision_action>\w): (?P<decision_price>[0-9.]+), price:[\s]*(?P<price>[0-9.]+), long:[\s]*(?P<long_count>[0-9.]+), \u2713 long:[\s]*(?P<profitable_long_count>[0-9.]+), ' + \
    r'\u2713 long pct:[\s]*(?P<profitable_long_pct>[0-9.]+)%, long profit pct:[\s]*(?P<mean_long_profit_pct>[0-9.]+)%'
end_str_2_pattern = r'Account Value:[\s]*(?P<account_value>[0-9.]+), accont growth:[\s]*(?P<account_growth>[\-]*[0-9.]+)%, ' + \
          r'stock growth:[\s]*(?P<stock_growth>[\-]*[0-9.]+)%, growth diff:[\s]*(?P<growth_diff>[\-]*[0-9.]+)%'

column_name_list = []
long_count_list = []
profitable_long_count_list = []
profitable_long_pct_list = []
mean_long_profit_pct_list = []
account_growth_list = []
stock_growth_list = []
growth_diff_list = []
loss_list = []

for i in range(len(block_str_lst)):
    match = re.search(end_str_1_pattern, end_strs_lst[i][0])

    # print(match.groups())

    # Extract the variables from the match object
    # decision = (match.group('decision_action'), int(match.group('decision_price'))) # this is only the last decision; useless.
    # price = float(match.group('price')) # same as above
    long_count = int(match.group('long_count'))
    profitable_long_count = int(match.group('profitable_long_count'))
    profitable_long_pct = float(match.group('profitable_long_pct'))
    mean_long_profit_pct = float(match.group('mean_long_profit_pct'))

    # Print the extracted variables
    # print('decision:', decision)
    # print('price:', price)
    # print('long_count:', long_count)
    # print('profitable_long_count:', profitable_long_count)
    # print('profitable_long_pct:', profitable_long_pct)
    # print('mean_long_profit_pct:', mean_long_profit_pct)

    match = re.match(end_str_2_pattern, end_strs_lst[i][1])

    # Get the variable values from the regex match
    # account_value = float(match.group('account_value'))
    account_growth = float(match.group('account_growth'))
    stock_growth = float(match.group('stock_growth'))
    growth_diff = float(match.group('growth_diff'))

    # Print the extracted variable values
    # print('account_value:', account_value)
    # print('account_growth:', account_growth)
    # print('stock_growth:', stock_growth)
    # print('growth_diff:', growth_diff)


    split_string = block_str_lst[i].split(":")
    column_name = split_string[1]
    # print(end_strs_lst[i][0])
    # print(end_strs_lst[i][1])
    loss = float(loss_lst[i])
    
    
    column_name_list.append('N/A')
    long_count_list.append(long_count)
    profitable_long_count_list.append(profitable_long_count)
    profitable_long_pct_list.append(profitable_long_pct)
    mean_long_profit_pct_list.append(mean_long_profit_pct)
    account_growth_list.append(account_growth)
    stock_growth_list.append(stock_growth)
    growth_diff_list.append(growth_diff)
    loss_list.append(loss)

data_dict = {
    'blocked_column': column_name_list,
    'long_count': long_count_list,
    'profitable_long_count': profitable_long_count_list,
    'profitable_long_pct': profitable_long_pct_list,
    'mean_long_profit_pct': mean_long_profit_pct_list,
    'account_growth': account_growth_list,
    'stock_growth': stock_growth_list,
    'growth_diff': growth_diff_list,
    'loss': loss_list
}

tmp = pd.DataFrame(data_dict)
df = pd.concat([df, tmp])
# print(df)
df_sorted = df.sort_values('account_growth')
print(df_sorted)


df.to_csv('../data/feature_efficacy_test_AAPL.csv', index=False)