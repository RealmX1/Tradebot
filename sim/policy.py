import numpy as np
import torch
import statistics
import random
import sys
from datetime import datetime
import pytz

dir_path = "../alpaca_api"

if dir_path not in sys.path:
    sys.path.append(dir_path)

from alpaca_trade import *
from alpaca.trading.enums import *
from alpaca.trading.requests import *
from alpaca.trading.client import *
from account import Account


class Policy(object):
    def __init__(self):
        pass

    def decide(self):
        # assumes that hist is of shape (hist_window, hist_features)
        # assumes that prediction is of shape (pred_window, pred_features), and that first of pred_feature is close price
        return None

class RandomPolicy(Policy):
    def __init__(self, account, buy_threshold = 0.005, allow_short = False, short_threshold = -0.005):
        super().__init__()
        self.account = account
        self.allow_short = allow_short
        self.short_threshold = short_threshold

        self.threshold = buy_threshold

        self.bought = False
        self.short = False

        self.long_count = 1
        self.profitable_long_count = 0
        self.short_count = 1
        self.profitable_short_count = 0
        self.prev_action_price = 0

        self.long_profit_pct_list = [0]
        self.short_profit_pct_list = [0]
    
    def decide(self, symbol:str, hist, price, weighted_prediction, col_names):
        result = ('n',0)
        prob = [0.02, 0.02, 0.96]
        choices = range(-1,2)

        pred = random.choices(choices, weights=prob, k=1)[0]

        if pred == 1:
            if self.bought == True:
                # print("Already bought!")
                return result # only invest 0.375 of all balance once.
            purchase_num = self.account.place_buy_max_order(symbol, price, 0, optimal=False)
            if purchase_num > 0:
                self.account.complete_buy_order(symbol, 0)
                # print(f'bought all! {purchase_num} shares at {price}$')
                result = ('b',purchase_num)
                self.bought = True
            else:
                print("BANKRUPT!")
                quit()
            #     self.account.cancel_buy_order(symbol, 0)
            #     result = ('n',0)
        elif pred == -1: # (and prediction[0][0] < 0)
            purchase_num = self.account.place_sell_max_order(symbol, price, 0)
            if purchase_num > 0:
                self.account.complete_sell_order(symbol, 0)
                # print(f'bought all! {purchase_num} shares at {price}$')
                result = ('s',purchase_num)
            else:
                self.account.cancel_sell_order(symbol,0)
                result = ('n',0)

            if self.allow_short:
                if weighted_prediction < self.short_threshold: # start short condition
                    if self.short == True:
                        return result
                    purchase_num = self.account.place_short_max_order(symbol, price, 0)
                    if purchase_num != 0:
                        self.account.complete_short_order(symbol, 0)
                        # print(f'sold all! {purchase_num} shares at {price}$')
                        result = ('s', result[1] + purchase_num)
                        self.short = True
                    else:
                        self.account.cancel_short_order(symbol,0)
                        # result not changed

        if result[0] == 'b':
            if self.short == True:
                profit_pct = (self.prev_action_price - price) / self.prev_action_price
                self.short_profit_pct_list.append(profit_pct)
                self.short_count += 1
                if self.prev_action_price > price:
                    self.profitable_short_count += 1
                self.short = False

            self.prev_action_price = price
        if result[0] == 's':

            if self.bought == True:
                profit_pct = (price - self.prev_action_price) / self.prev_action_price * 100
                self.long_profit_pct_list.append(profit_pct)
                self.long_count += 1
                if self.prev_action_price < price:
                    self.profitable_long_count += 1
                self.bought = False

            self.prev_action_price = price

        result = (result[0], int(result[1]))
        return result

    def get_trade_stat(self):
        # print("long profit pct list: ", self.long_profit_pct_list)
        # print("short profit pct list: ", self.short_profit_pct_list)
        mean_long_profit_pct = np.mean(self.long_profit_pct_list)
        mean_short_profit_pct = np.mean(self.short_profit_pct_list)
        return self.long_count, self.profitable_long_count, self.short_count, self.profitable_short_count, mean_long_profit_pct, mean_short_profit_pct

            
            

class SimpleLongShort(Policy):
    def __init__(self, account, buy_threshold = 0.005, allow_short = False, short_threshold = -0.005):
        super().__init__()
        self.account = account
        self.allow_short = allow_short
        self.short_threshold = short_threshold

        self.threshold = buy_threshold

        self.bought = False
        self.short = False

        self.long_count = 1
        self.profitable_long_count = 0
        self.short_count = 1
        self.profitable_short_count = 0
        self.prev_action_price = 0

        self.long_profit_pct_list = [0]
        self.short_profit_pct_list = [0]
    
    def decide(self, symbol:str, hist, price, weighted_prediction, col_names):
        
        last_hist = hist[0,-1,:]

        edt_scale_col = locate_cols(col_names, 'edt_scaled')[0]
        # print(edt_scale_col)

        edt_scale = last_hist[edt_scale_col]
        # print(edt_scale)

        
        
        # Simple policy
        if weighted_prediction > 0:
            pred = 1
        else:
            pred = -1

        if weighted_prediction < self.threshold and weighted_prediction > -self.threshold:
            pred = 0

        # if edt_scale < -100:
        #     return result
        if edt_scale > 1.6:
            pred = -2 # end of day signal

        
        result = ('n',0) # by default don't do anything

        if pred == 1:
            if self.bought == True:
                # print("Already bought!")
                return result # only invest 0.375 of all balance once.
            purchase_num = self.account.place_buy_max_order(symbol, price, 0, optimal=False)
            if purchase_num > 0:
                self.account.complete_buy_order(symbol, 0)
                # print(f'bought all! {purchase_num} shares at {price}$')
                result = ('b',purchase_num)
                self.bought = True
            else:
                print("BANKRUPT!")
                quit()
            #     self.account.cancel_buy_order(symbol, 0)
            #     result = ('n',0)
        elif pred == -1: # (and prediction[0][0] < 0)
            purchase_num = self.account.place_sell_max_order(symbol, price, 0)
            if purchase_num > 0:
                self.account.complete_sell_order(symbol, 0)
                # print(f'bought all! {purchase_num} shares at {price}$')
                result = ('s',purchase_num)
            else:
                self.account.cancel_sell_order(symbol,0)
                result = ('n',0)

            if self.allow_short:
                if weighted_prediction < self.short_threshold: # start short condition
                    if self.short == True:
                        return result
                    purchase_num = self.account.place_short_max_order(symbol, price, 0)
                    if purchase_num != 0:
                        self.account.complete_short_order(symbol, 0)
                        # print(f'sold all! {purchase_num} shares at {price}$')
                        result = ('s', result[1] + purchase_num)
                        self.short = True
                    else:
                        self.account.cancel_short_order(symbol,0)
                        # result not changed
        elif pred == -2: # end all short position/sell all long position at the end of day.
            purchase_num = self.account.place_reverse_short_order(symbol, price,0)
            if purchase_num > 0: # has short position to reverse.
                self.account.complete_reverse_short_order(symbol, 0)
                # print(f'reversed short! {purchase_num} shares at {price}$')
                result = ('b', purchase_num)
            elif purchase_num == 0:
                # purchase_num = self.account.cancel_reverse_short_order(symbol, 0) # no buy order is placed
                # result is default result

                # since no need to reverse short position -- we might have positive hold position; sell all of them.
                purchase_num = self.account.place_sell_max_order(symbol, price, 0)
                if purchase_num > 0:
                    self.account.complete_sell_order(symbol, 0)
                    # print(f'bought all! {purchase_num} shares at {price}$')
                    result = ('s',purchase_num)
                else:
                    self.account.cancel_sell_order(symbol,0)
                    # result = ('n',0)
                    # result is as default
            elif purchase_num < 0:
                print("bankrupt!")
                quit()

        if result[0] == 'b':
            if self.short == True:
                profit_pct = (self.prev_action_price - price) / self.prev_action_price
                self.short_profit_pct_list.append(profit_pct)
                self.short_count += 1
                if self.prev_action_price > price:
                    self.profitable_short_count += 1
                self.short = False

            self.prev_action_price = price
        if result[0] == 's':

            if self.bought == True:
                profit_pct = (price - self.prev_action_price) / self.prev_action_price * 100
                self.long_profit_pct_list.append(profit_pct)
                self.long_count += 1
                if self.prev_action_price < price:
                    self.profitable_long_count += 1
                self.bought = False

            self.prev_action_price = price

        result = (result[0], int(result[1]))
        return result

    def get_trade_stat(self):
        # print("long profit pct list: ", self.long_profit_pct_list)
        # print("short profit pct list: ", self.short_profit_pct_list)
        mean_long_profit_pct = np.mean(self.long_profit_pct_list)
        mean_short_profit_pct = np.mean(self.short_profit_pct_list)
        return self.long_count, self.profitable_long_count, self.short_count, self.profitable_short_count, mean_long_profit_pct, mean_short_profit_pct

    def has_long_position(self):
        return self.bought or self.short

def locate_cols(strings_list, substring):
    return [i for i, string in enumerate(strings_list) if substring in string]



##############################################################################################################################
class NaiveLongShort(Policy):
    def __init__(self, account):
        super().__init__()
        self.account = account
        self.threshold = 0.01

        self.bought = False
        self.short = False

        self.long_count = 1
        self.profitable_long_count = 0
        self.short_count = 1
        self.profitable_short_count = 0
        self.prev_action_price = 0

        self.long_profit_pct_list = [0]
        self.short_profit_pct_list = [0]
    
    def decide(self, symbol, price, action):
        # print(prediction.shape)
        result = ('n',0)
        
        
        if action == 0:
            if self.bought == True:
                # print("Already bought!")
                return result # only invest 0.375 of all balance once.
            purchase_num = self.account.place_buy_max_order(symbol, price, 0, optimal=False)
            if purchase_num > 0:
                self.account.complete_buy_order(symbol, 0)
                # print(f'bought all! {purchase_num} shares at {price}$')
                result = ('b',purchase_num)
                self.bought = True
            # else:
            #     print("BANKRUPT!")
            #     quit()
            #     self.account.cancel_buy_order(symbol, 0)
            #     result = ('n',0)
        elif action == 1 or action == 2: # (and prediction[0][0] < 0)
            purchase_num = self.account.place_sell_max_order(symbol, price, 0)
            if purchase_num > 0:
                self.account.complete_sell_order(symbol, 0)
                # print(f'bought all! {purchase_num} shares at {price}$')
                result = ('s',purchase_num)
            else:
                self.account.cancel_sell_order(symbol,0)
                # result = ('n',0)


            if action == 2: # start short condition
                if self.short == True:
                    return result
                purchase_num = self.account.place_short_max_order(symbol, price, 0)
                if purchase_num != 0:
                    self.account.complete_short_order(symbol, 0)
                    # print(f'sold all! {purchase_num} shares at {price}$')
                    result = ('s', result[1] + purchase_num)
                    self.short = True
                else:
                    self.account.cancel_short_order(symbol,0)
                    # result not changed

        elif action == 3: # end all short position/sell all long position at the end of day.
            purchase_num = self.account.place_reverse_short_order(symbol, price,0)
            if purchase_num > 0:
                self.account.complete_reverse_short_order(symbol, 0)
                # print(f'reversed short! {purchase_num} shares at {price}$')
                result = ('b', purchase_num)
            elif purchase_num == 0:
                # purchase_num = self.account.cancel_reverse_short_order(symbol, 0) # no buy order is placed
                # result is default result

                # since no need to reverse short position -- we might have positive hold position; sell all of them.
                purchase_num = self.account.place_sell_max_order(symbol, price, 0)
                if purchase_num > 0:
                    self.account.complete_sell_order(symbol, 0)
                    # print(f'bought all! {purchase_num} shares at {price}$')
                    result = ('s',purchase_num)
                else:
                    self.account.cancel_sell_order(symbol,0)
                    # result = ('n',0)
                    # result is as default
            elif purchase_num < 0:
                print("bankrupt!")
                quit()

        if result[0] == 'b':
            if self.short == True:
                profit_pct = (self.prev_action_price - price) / self.prev_action_price
                self.short_profit_pct_list.append(profit_pct)
                self.short_count += 1
                if self.prev_action_price > price:
                    self.profitable_short_count += 1
                self.short = False

            self.prev_action_price = price
        if result[0] == 's':

            if self.bought == True:
                profit_pct = (price - self.prev_action_price) / self.prev_action_price * 100
                self.long_profit_pct_list.append(profit_pct)
                self.long_count += 1
                if self.prev_action_price < price:
                    self.profitable_long_count += 1
                self.bought = False

            self.prev_action_price = price

        result = (result[0], int(result[1]))
        return result

    def get_trade_stat(self):
        # print("long profit pct list: ", self.long_profit_pct_list)
        # print("short profit pct list: ", self.short_profit_pct_list)
        mean_long_profit_pct = statistics.mean(self.long_profit_pct_list)
        mean_short_profit_pct = statistics.mean(self.short_profit_pct_list)
        return self.long_count, self.profitable_long_count, self.short_count, self.profitable_short_count, mean_long_profit_pct, mean_short_profit_pct



#######################################################################################################################
class AlpacaSimpleLong(Policy):
    def __init__(self, trading_client, buy_threshold = 0.005, allow_short = False, short_threshold = -0.005, ):
        super().__init__()
        # probably should get all orders from the account first...
        self.trading_client = trading_client
        self.allow_short = allow_short
        self.short_threshold = short_threshold

        self.threshold = buy_threshold

        self.bought = {}
        self.short = {}
        self.position = {}

        
        # real-time info
        self.info_dict = {}
        self.edt_scale = 0
        self.edt_scale_threshold = 1.6

        # orders
        self.orders = []
        self.order_overtime_limit = 15

        # statistics
        self.long_count = 1
        self.profitable_long_count = 0
        self.short_count = 1
        self.profitable_short_count = 0
        self.prev_action_price = {}

        self.long_profit_pct_list = [0]
        self.short_profit_pct_list = [0]


        

        positions = self.trading_client.get_all_positions()
        for position in positions:
            symbol = position.symbol
            price = position.avg_entry_price
            shares = position.qty
            side = position.side

            self.prev_action_price[symbol] = price
            if side == PositionSide.LONG:
                self.bought[symbol] = True
                self.short[symbol] = False
            else:
                self.bought[position.symbol] = False
                self.short[position.symbol] = True
            
            self.position[symbol] = {'side':side, 'shares':shares, 'price':price}
        
        self.top_n = self.bottom_n = 2 # how many symbols to scan for buy and short -- NOT INTENDED TO LIMIT NORMAL SELL
        self.single_symbol_limit = 10000000
        self.single_symbol_limit_pct = 0.1

        self.single_transaction_limit_pct = 0.0001 # with respect to volume?
        self.single_transaction_limit = 10000000

    def process(self, symbol, hist, weighted_prediction, col_lst):
        last_hist = hist[0,-1,:]

        price_col = col_lst.index('close')
        edt_scale_col = col_lst.index('edt_scaled')

        self.edt_scale = last_hist[edt_scale_col]

        price = last_hist[price_col]
        
        self.info_dict[symbol] = {'price':price, 'weighted_prediction':weighted_prediction}

    
    def decide_sell(self, bottom_n_pred):
        for symbol, info in bottom_n_pred: # decide if to SELL (or short?)
            price = info['price']
            pred = info['weighted_prediction']
            if pred > - self.self_threshold: # no need to continue if current lowest is above threshold.
                break
            
            # check if continued selling is allowed
            position_side = None
            position_value = 0
            has_position = (symbol in self.position.keys())
            if has_position and position_side == PositionSide.LONG:
                position_side = self.position[symbol]['side']
                position_value = self.position[symbol]['shares'] * price
            else:
                #short?
                continue
            
            sell_num = max(0, min(self.position[symbol]['shares'],
                                  self.single_transaction_limit // price
                                  # self.single_transaction_limit_pct * stock volumn # no need to divide price here.
            ))

            if sell_num == 0:
                print("something might have gone wrong; you have a position, yet you can't sell it.")

            limit_sell_order = create_limit_order(symbol, sell_num, price, order_side = OrderSide.sell, tif = TimeInForce.GTC, stop = None)
            order = self.trading_client.submit_order(limit_sell_order)

            self.orders.append(order.id)
            # buying_power -= purchase_value # short; frozen buying power that can only be used to buy back same stock.
            # self.short[symbol] = True
    
    def decide_buy(self, top_n_pred):
        single_symbol_portfolio_pct_limit_value = self.trading_client.get_account().portfolio_value * self.single_symbol_limit_pct
        for symbol, info in top_n_pred: # decide if to LONG (note that reverse short isn't considered here... or should it? -- IT SHOULD)
            price = info['price']
            pred = info['weighted_prediction']
            if pred < self.threshold: # no need to continue if current highest is below threshold.
                break
            
            # check if continued purchase is allowed
            position_side = None
            position_value = 0
            has_position = (symbol in self.position.keys())
            if has_position:
                position_side = self.position[symbol]['side']
                position_value = self.position[symbol]['shares'] * price
                
            purchase_value = max(0, min(buying_power, 
                                    single_symbol_portfolio_pct_limit_value - position_value, # how wil this work on shorts??? \ 
                                    self.single_symbol_limit - position_value,
                                    self.single_transaction_limit
                                    ),
                                )
            purchase_num = purchase_value // price

            assert purchase_num >= 0
            if purchase_num == 0: # NOT POSSIBLE TO GO UDNER=]
                continue
            # might need more complex logic to determine order price; but decision model is also an option (that can't utilize continued development on this side), so...
            limit_buy_order = create_limit_order(symbol, purchase_num, price, order_side = OrderSide.BUY, tif = TimeInForce.GTC, stop = None)
            order = self.trading_client.submit_order(limit_buy_order)
            
            self.orders.append(order.id)
            buying_power -= purchase_value
            self.bought[symbol] = True

    def decide(self): # clean self.info() at the end of decide()?
            
        sorted_info = sorted(self.info_dict.items(), key=lambda x: x[1]['weighted_prediction'], reverse=True)
        top_n_pred = sorted_info[:self.top_n]
        bottom_n_pred = sorted_info[-self.bottom_n:].reverse() # reverse to let this start from lowest pred
        
        alpaca_account = self.trading_client.get_account()
        buying_power = float(alpaca_account.buying_power)
        portfolio_value = float(alpaca_account.portfolio_value)

        self.decide_sell(self, bottom_n_pred)

        if self.edt_scale < self.edt_scale_threshold:
            self.decide_buy(self, top_n_pred, alpaca_account)

        
        


        



        # positions = []
        # for symbol in self.info_dict.keys():
        #     positions.append(self.trading_client.get_open_position(symbol))
        returns = []
        result = ('n',0) # by default don't do anything


        if pred > 0:
            signal = 1
        else:
            signal = -1

        if pred < self.threshold and pred > -self.threshold:
            signal = 0

        # if edt_scale < -100:
        #     return result
        if self.edt_scale > 1.6:
            signal = -2 # end of day signal

        buying_power = float(alpaca_account.buying_power)


        if signal == 1:
            # buy in
            pass
        elif signal == -1: # (and prediction[0][0] < 0)
            
            # has position to sell
            if self.position.has_key(symbol) == False:
                print("No active long holding on this stock")
                return result
            sell_num = self.position[symbol]['shares']
            if sell_num > 0:
                
                # print(f'bought all! {purchase_num} shares at {price}$')
                result = ('s',sell_num)
            else:
                self.account.cancel_sell_order(symbol,0)
                result = ('n',0)

            # if self.allow_short:
            #     if weighted_prediction < self.short_threshold: # start short condition
            #         if self.short == True:
            #             return result
            #         purchase_num = self.account.place_short_max_order(symbol, price, 0)
            #         if purchase_num != 0:
            #             pass
            #             # self.account.complete_short_order(symbol, 0)
            #             # # print(f'sold all! {purchase_num} shares at {price}$')
            #             # result = ('s', result[1] + purchase_num)
            #             # # self.short = True
            #         else:
            #             self.account.cancel_short_order(symbol,0)
            #             # result not changed
        elif signal == -2: # end all short position/sell all long position at the end of day.
            purchase_num = 0 # self.account.place_reverse_short_order(symbol, price,0)
            if purchase_num > 0:
                print('has short position to reverse!')
                pass
                # self.account.complete_reverse_short_order(symbol, 0)
                # # print(f'reversed short! {purchase_num} shares at {price}$')
                # result = ('b', purchase_num)
            elif purchase_num == 0:
                print('no short position to reverse')
                print('closing all long positions:')
                # purchase_num = self.account.cancel_reverse_short_order(symbol, 0) # no buy order is placed
                # result is default result

                # since no need to reverse short position -- we might have positive hold position; sell all of them.

                if has_long_position:
                    sell_num = 0
                    # self.account.complete_sell_order(symbol, 0)
                    # # print(f'bought all! {purchase_num} shares at {price}$')
                    # result = ('s',purchase_num)
                else:
                    self.account.cancel_sell_order(symbol,0)
                    # result = ('n',0)
                    # result is as default
            elif purchase_num < 0:
                print("bankrupt!")
                quit()

        self.info_dict = {}
        
        return result
    def update_account_status(self):
        now = datetime.utcnow().replace(tzinfo=pytz.utc)
        finished_orders = []
        for order_id in self.orders:
            order = self.trading_client.get_order_by_id(order_id)
            print(f'order id: {order_id}')
            print(order.status)
            if order.status == OrderStatus.FILLED:
                if order.side == OrderSide.BUY:
                    self.complete_buy(order.symbol)
                elif order.side == OrderSide.SELL:
                    self.complete_sell(order.symbol)
                finished_orders.append(order_id)
            elif order.status == OrderStatus.PARTIALLY_FILLED:
                if (now-order.filled_at).total_seconds() > self.order_overtime_limit or\
                    (now-order.created_at).total_seconds() > 30:
                    print("OVERTIME!")
                    self.trading_client.cancel_order(order_id)
                    if order.side == OrderSide.BUY:
                        self.cancel_buy(order.symbol)
                    elif order.side == OrderSide.SELL:
                        self.cancel_sell(order.symbol)
                    finished_orders.append(order_id)
            elif order.status == OrderStatus.ACCEPTED:
                if (now-order.created_at).total_seconds() > self.order_overtime_limit:
                    print("OVERTIME!")
                    self.trading_client.cancel_order(order_id)
                    if order.side == OrderSide.BUY:
                        self.cancel_buy(order.symbol)
                    elif order.side == OrderSide.SELL:
                        self.cancel_sell(order.symbol)
                    finished_orders.append(order_id)
            
            # for property_name, value in order:
            #     print(f'"{property_name}": {value}')
        
        if len(finished_orders) > 0:
            print(f'finished orders: {finished_orders}')
            self.orders.remove(finished_orders)

    def update_position(self):
        positions = self.trading_client.list_positions()
        for position in positions:
            symbol = position.symbol
            # if position.side == PositionSide.LONG:
            #     self.bought[symbol] = True
            # elif position.side == PositionSide.SHORT:
            #     self.short[symbol] = True
            self.position[symbol] = (position.side, position.quantity, position.avg_entry_price)
    
    def complete_buy(self, order):
        symbol = order.symbol
        shares = order.filled_quantity
        price = order.filled_avg_price

        prev_action_price = self.prev_action_price[symbol]

        if self.short[symbol] == True:
            profit_pct = (prev_action_price - price) / prev_action_price
            self.short_profit_pct_list.append(profit_pct)
            self.short_count += 1
            # self.short_total_volume
            # self.short_total_profit (not profitpct aggregate)
            # need massive overhaul to intorduce short function
            if prev_action_price > price:
                self.profitable_short_count += 1
            self.short[symbol] = False

        self.prev_action_price[symbol] = price

        return ('b', shares)

    def cancel_buy(self, order):
        if order.status == OrderStatus.PARTIALLY_FILLED:
            

        symbol = order.symbol

        self.bought[symbol] = False
        self.trading_client.cancel_order(order.id)
    
    def complete_sell(self, order):
        symbol = order.symbol
        shares = order.filled_quantity
        price = order.filled_avg_price

        prev_action_price = self.prev_action_price[symbol]

        if self.bought[symbol] == True:
            profit_pct = (price - prev_action_price) / prev_action_price * 100
            self.long_profit_pct_list.append(profit_pct)
            self.long_count += 1
            # self.long_total_spent += prev_action_price * shares
            # self.long_total_profit += shares(price - prev_action_price)
            if prev_action_price < price:
                self.profitable_long_count += 1
            self.bought[symbol] = False

        self.prev_action_price[symbol] = price

        return ('s', shares)
    
    def cancel_sell(self, order):
        # symbol = order.symbol
        # self.short[symbol] = False # ???????? the logic behind this is more complex. not all sells are shorts... or is it really more complicated?
        self.trading_client.cancel_order(order.id)

    def get_trade_stat(self):
        # print("long profit pct list: ", self.long_profit_pct_list)
        # print("short profit pct list: ", self.short_profit_pct_list)
        mean_long_profit_pct = np.mean(self.long_profit_pct_list)
        mean_short_profit_pct = np.mean(self.short_profit_pct_list)
        return self.long_count, self.profitable_long_count, self.short_count, self.profitable_short_count, mean_long_profit_pct, mean_short_profit_pct


def locate_cols(strings_list, substring):
    return [i for i, string in enumerate(strings_list) if substring in string]


if __name__ == '__main__':
    # things to test: start with no position, start with some position;
    # buy, sell, overtime buy, overtime, sell, partial buy, partial sell
    import pickle
    import os

    trading_client = TradingClient(PAPER_API_KEY, PAPER_SECRET_KEY, paper = True)
    # print(trading_client.get_open_positions(''))

    p = AlpacaSimpleLong(trading_client)
    
    # if os.path.exists("my_object.pkl"):
    #     print('loading existing policy object')
    #     with open("my_object.pkl", "rb") as f:
    #         p = pickle.load(f)
    #     print(p.orders)
    p.update_account_status()

    hist = np.ones((1,30, 16))
    hist[:,:,0] = 200
    col_lst = ['close','edt_scaled']
    # last_hist = hist[0,-1,:]
    p.process('AAPL', hist, 1.0, col_lst)
    decision = p.decide()
    print(decision)

    p.update_account_status()


    
    # with open("my_object.pkl", "wb") as f:
    #     pickle.dump(p, f)
    # long = SimpleLongShort()
    # acc = Account(100000, ['BABA'])
    
    # price = 100
    # prediction = np.arange(101,111).reshape(10,1)
    
    # decision = long.decide('BABA', None, price, prediction, acc)
    # print(decision)
    # print(acc.holding)





















# class PolicyComplex(Policy):
#     def __init__(self, hist_window, hist_features, pred_window, pred_features):
#         super().__init__(hist_window, hist_features, pred_window, pred_features)
    
#     def decide(self, symbol:str, hist, price, prediction, account):
#         pass

#     # TODO: cancel order if waiting too long (after 20s?)
