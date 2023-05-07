import pickle
import re
import numpy as np
import pandas as pd

with open('lists_no_multithread_MSFT_block.pkl', 'rb') as f:
    loaded_lists = pickle.load(f)

block_str_lst, test_strs_lst, loss_lst = loaded_lists

# with open('lists_no_multithread_AAPL_noblock.pkl', 'rb') as f:
#     loaded_lists = pickle.load(f)
# # block_str_lst, test_strs_lst, loss_lst = loaded_lists
# block_str_lst2, test_strs_lst2, loss_lst2 = loaded_lists
# block_str_lst += block_str_lst2
# test_strs_lst += test_strs_lst2
# loss_lst += loss_lst2

# df = pd.read_csv('../data/feature_efficacy_test_AAPL.csv')
print(len(block_str_lst))

'''
prediction_stat_str = \
    f'decision: {decision[0]}:{decision[1]:>5}, ' + \
    f'price: {price:>6.2f}, ' + \
    f'unfilled buy & sell: {unfilled_buy:>4}, {unfilled_sell:>4}, ' + \
    f'long: {long_count:>4}, ' + \
    f'\u2713 long: {profitable_long_count:>4}, ' + \
    f'\u2713 long pct: {profitable_long_count/long_count*100:>5.2f}%, ' + \
    f'long profit pct: {mean_long_profit_pct:>6.4f}%, ' + \
    f'occupancy rate: {zero_balance_timer/(i+1)*100:>5.2f}%, '
'''

prediction_stat_str_pattern = (
    r'decision:[\s]*(?P<decision_action>\w):[\s]*(?P<decision_price>[0-9.]+), '
    r'price:[\s]*(?P<price>[0-9.]+), '
    r'unfilled buy & sell:[\s]*(?P<unfilled_buy>[0-9]+),[\s]*(?P<unfilled_sell>[0-9]+), '
    r'long:[\s]*(?P<long_count>[0-9.]+), '
    r'\u2713 long:[\s]*(?P<profitable_long_count>[0-9.]+), '
    r'\u2713 long pct:[\s]*(?P<profitable_long_pct>[0-9.]+)%, '
    r'long profit pct:[\s]*(?P<mean_long_profit_pct>[0-9.]+)%, '
    r'occupancy rate:[\s]*(?P<occupancy_rate>[0-9.]+)%'
)

'''
account_n_stock_str = \
    f'Account Value: {account_value:>10.2f}, ' + \
    f'accont growth: {account_growth:>6.2f}%, ' + \
    f'stock growth: {stock_growth:>6.2f}%, ' +  \
    f'pct growth diff: {pct_growth_diff:>6.2f}%, ' + \
    f'interval per trade: {i/(long_count+short_count):>4.2f}, ' + \
    f'i/t since last plot: {(i-prev_interval)/(long_count+short_count-prev_long_count-prev_short_count + 1):>4.2f}, ' #+ \
'''

account_n_stock_str_pattern = (
    r'Account Value:[\s]*(?P<account_value>[0-9.]+), '
    r'accont growth:[\s]*(?P<account_growth>[0-9.]+)%, '
    r'stock growth:[\s]*(?P<stock_growth>[0-9.]+)%, '
    r'pct growth diff:[\s]*(?P<pct_growth_diff>[0-9.]+)%, '
    r'interval per trade:[\s]*(?P<interval_per_trade>[0-9.]+), '
    r'i/t since last plot:[\s]*(?P<it_since_last_plot>[0-9.]+),'
)



columns = [
    'column_name', 'unfilled_buy', 'unfilled_sell', 'long_count',
    'profitable_long_count', 'profitable_long_pct', 'mean_long_profit_pct',
    'occupancy_rate', 'account_value', 'account_growth', 'stock_growth',
    'pct_growth_diff', 'interval_per_trade', 'it_since_last_plot', 'loss'
]
df = pd.DataFrame(columns=columns)

for i in range(len(block_str_lst)):
    block_str = block_str_lst[i]
    prediction_stat_str, account_n_stock_str = test_strs_lst[i]
    print(prediction_stat_str)
    loss = float(loss_lst[i])
    # print(test_strs_lst[i][0])

    split_string = block_str.split(":")
    column_name = split_string[1]

    match = re.search(prediction_stat_str_pattern, prediction_stat_str)
    # decision_action = match.group('decision_action')
    # decision_price = float(match.group('decision_price'))
    # price = float(match.group('price'))
    unfilled_buy = int(match.group('unfilled_buy'))
    unfilled_sell = int(match.group('unfilled_sell'))
    long_count = int(match.group('long_count'))
    profitable_long_count = int(match.group('profitable_long_count'))
    profitable_long_pct = float(match.group('profitable_long_pct'))
    mean_long_profit_pct = float(match.group('mean_long_profit_pct'))
    occupancy_rate = float(match.group('occupancy_rate'))

    # Print the extracted variables
    # print('decision_action:', decision_action)
    # print('decision_price:', decision_price)
    # print('price:', price)
    print('unfilled_buy & _sell:', unfilled_buy, unfilled_sell)
    print('long_count:', long_count)
    print('profitable_long_count:', profitable_long_count)
    print('profitable_long_pct:', profitable_long_pct)
    print('mean_long_profit_pct:', mean_long_profit_pct)
    print('occupancy_rate:', occupancy_rate)


##############################################################################
    match = re.search(account_n_stock_str_pattern, account_n_stock_str)
    account_value = float(match.group('account_value'))
    account_growth = float(match.group('account_growth'))
    stock_growth = float(match.group('stock_growth'))
    pct_growth_diff = float(match.group('pct_growth_diff'))
    interval_per_trade = float(match.group('interval_per_trade'))
    it_since_last_plot = float(match.group('it_since_last_plot'))

    # Print the extracted variables
    print('account_value:', account_value)
    print('account_growth:', account_growth)
    print('stock_growth:', stock_growth)
    print('pct_growth_diff:', pct_growth_diff)
    print('interval_per_trade:', interval_per_trade)
    print('it_since_last_plot:', it_since_last_plot)

    loss = float(loss_lst[i])
    

    data = {
        'column_name': column_name,
        # 'unfilled_buy': unfilled_buy,
        # 'unfilled_sell': unfilled_sell,
        'long_count': long_count,
        'profitable_long_count': profitable_long_count,
        'profitable_long_pct': profitable_long_pct,
        'mean_long_profit_pct': mean_long_profit_pct,
        'occupancy_rate': occupancy_rate,
        'account_value': account_value,
        'account_growth': account_growth,
        'stock_growth': stock_growth,
        'pct_growth_diff': pct_growth_diff,
        'interval_per_trade': interval_per_trade,
        'it_since_last_plot': it_since_last_plot,
        # 'loss': loss
    }
    df = df.append(data, ignore_index=True)
df.dropna(axis=1, how='all', inplace=True)
print(df)
# print(tmp)
# df = pd.concat([df, tmp])
# print(df)
df_sorted = df.sort_values('account_growth')
print(df_sorted)


# df_sorted.to_csv('../data/feature_efficacy_test_AAPL_FULL.csv', index=False)







'''
GPT-4 instructions:
I have a string created through the following code:

prediction_stat_str = \
                            f'decision: {decision[0]}:{decision[1]:>5}, ' + \
                            f'price: {price:>6.2f}, ' + \
                            f'unfilled buy & sell: {unfilled_buy:>4}, {unfilled_sell:>4}, ' + \
                            f'long: {long_count:>4}, ' + \
                            f'\u2713 long: {profitable_long_count:>4}, ' + \
                            f'\u2713 long pct: {profitable_long_count/long_count*100:>5.2f}%, ' + \
                            f'long profit pct: {mean_long_profit_pct:>6.4f}%, ' + \
                            f'occupancy rate: {zero_balance_timer/(i+1)*100:>5.2f}%, '

I want to extract all of those formated information (i.e., price, unfilled_buy, ... zero_balance_timer/(i+1)*100) from such a string.


Here is an example code that worked for a different piece of string that is constructed in a similar way:

import re
end_str_1_pattern = r'decision:[\s]*(?P<decision_action>\w):[\s]*(?P<decision_price>[0-9.]+), price:[\s]*(?P<price>[0-9.]+), long:[\s]*(?P<long_count>[0-9.]+), \u2713 long:[\s]*(?P<profitable_long_count>[0-9.]+), ' + \
    r'\u2713 long pct:[\s]*(?P<profitable_long_pct>[0-9.]+)%, long profit pct:[\s]*(?P<mean_long_profit_pct>[0-9.]+)%'

match = re.search(end_str_1_pattern, end_str)
long_count = int(match.group('long_count'))
profitable_long_count = int(match.group('profitable_long_count'))
profitable_long_pct = float(match.group('profitable_long_pct'))
mean_long_profit_pct = float(match.group('mean_long_profit_pct'))
'''