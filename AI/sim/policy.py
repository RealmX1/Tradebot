from .account import Account
import numpy as np
import torch
import statistics
import random

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
        # print(prediction.shape)
        # prediction_window = prediction.shape[0]

        # weight_decay = 0.2
        # arr = np.ones(prediction_window)
        # for i in range(1, prediction_window):
        #     arr[i] = arr[i-1] * weight_decay
        # weights = arr.reshape(prediction_window,1)
        
        # weighted_prediction = (prediction * weights).sum() / weights.sum()

        # print(f'prediction: {prediction}, weighted_prediction: {weighted_prediction}')
        
        last_hist = hist[0,-1,:]

        edt_scale_col = locate_cols(col_names, 'edt_scaled')[0]
        # print(edt_scale_col)

        edt_scale = last_hist[edt_scale_col]
        # print(edt_scale)

        result = ('n',0) # by default don't do anything


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
        mean_long_profit_pct = np.mean(self.long_profit_pct_list)
        mean_short_profit_pct = np.mean(self.short_profit_pct_list)
        return self.long_count, self.profitable_long_count, self.short_count, self.profitable_short_count, mean_long_profit_pct, mean_short_profit_pct

    def has_position(self):
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


class PolicyComplex(Policy):
    def __init__(self, hist_window, hist_features, pred_window, pred_features):
        super().__init__(hist_window, hist_features, pred_window, pred_features)
    
    def decide(self, symbol:str, hist, price, prediction, account):
        pass

    # TODO: cancel order if waiting too long (after 20s?)

class ReinforcementPolicy(Policy):
    # return anything that may be considered a state for the RL.
    pass
















#######################################################################################################################
class AlpacaSimpleLongShort(Policy):
    def __init__(self, buy_threshold = 0.005, allow_short = False, short_threshold = -0.005):
        super().__init__()
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
    
    def decide(self, symbol:str, hist, price, weighted_prediction, col_names, alpaca_account):
        last_hist = hist[0,-1,:]

        edt_scale_col = locate_cols(col_names, 'edt_scaled')[0]
        # print(edt_scale_col)

        edt_scale = last_hist[edt_scale_col]
        # print(edt_scale)

        result = ('n',0) # by default don't do anything


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

        
        if pred == 1:
            if self.bought == True:
                # print("Already bought!")
                return result # only invest 0.375 of all balance once if optimal is on.
            purchase_num = self.account.place_buy_max_order(symbol, price, 0, optimal=False)
            
            self.bought = True
            if purchase_num > 0:
                self.account.complete_buy_order(symbol, 0)
                # print(f'bought all! {purchase_num} shares at {price}$')
                result = ('b',purchase_num)
                # self.bought = True
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
                        pass
                        # self.account.complete_short_order(symbol, 0)
                        # # print(f'sold all! {purchase_num} shares at {price}$')
                        # result = ('s', result[1] + purchase_num)
                        # # self.short = True
                    else:
                        self.account.cancel_short_order(symbol,0)
                        # result not changed
        elif pred == -2: # end all short position/sell all long position at the end of day.
            purchase_num = self.account.place_reverse_short_order(symbol, price,0)
            if purchase_num > 0:
                pass
                # self.account.complete_reverse_short_order(symbol, 0)
                # # print(f'reversed short! {purchase_num} shares at {price}$')
                # result = ('b', purchase_num)
            elif purchase_num == 0:
                # purchase_num = self.account.cancel_reverse_short_order(symbol, 0) # no buy order is placed
                # result is default result

                # since no need to reverse short position -- we might have positive hold position; sell all of them.
                purchase_num = self.account.place_sell_max_order(symbol, price, 0)
                if purchase_num > 0:
                    pass
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

        return result
    
    def complete_buy(self, symbol):
        shares = self.account.complete_buy_order(symbol, 0)

        if self.short == True:
            profit_pct = (self.prev_action_price - price) / self.prev_action_price
            self.short_profit_pct_list.append(profit_pct)
            self.short_count += 1
            if self.prev_action_price > price:
                self.profitable_short_count += 1
            self.short = False

        self.prev_action_price = price

        return ('b', shares)

    def cancel_buy(self, symbol):
        self.account.cancel_buy_order(symbol, 0)
    
    def complete_sell(self, symbol):
        shares = self.account.complete_sell_order(symbol, 0)

        if self.bought == True:
            profit_pct = (price - self.prev_action_price) / self.prev_action_price * 100
            self.long_profit_pct_list.append(profit_pct)
            self.long_count += 1
            if self.prev_action_price < price:
                self.profitable_long_count += 1
            self.bought = False

        self.prev_action_price = price

        return ('s', shares)
    
    def cancel_sell(self, symbol):
        self.account.cancel_sell_order(symbol, 0)
    
    # def complete_short(self, symbol):
    #     self.short = True
    #     shares = self.account.complete_short_order(symbol, 0)
    #     return ('s', shares)
    
    # def cancel_short(self, symbol):
    
    

    def get_trade_stat(self):
        # print("long profit pct list: ", self.long_profit_pct_list)
        # print("short profit pct list: ", self.short_profit_pct_list)
        mean_long_profit_pct = np.mean(self.long_profit_pct_list)
        mean_short_profit_pct = np.mean(self.short_profit_pct_list)
        return self.long_count, self.profitable_long_count, self.short_count, self.profitable_short_count, mean_long_profit_pct, mean_short_profit_pct


def locate_cols(strings_list, substring):
    return [i for i, string in enumerate(strings_list) if substring in string]


if __name__ == '__main__':
    long = SimpleLongShort()
    acc = Account(100000, ['BABA'])
    
    price = 100
    prediction = np.arange(101,111).reshape(10,1)
    
    decision = long.decide('BABA', None, price, prediction, acc)
    print(decision)
    print(acc.holding)