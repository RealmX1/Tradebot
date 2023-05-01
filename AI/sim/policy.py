from .account import Account
import numpy as np
import torch
import statistics

class Policy(object):
    def __init__(self):
        pass

    def decide(self, hist, price, prediction, account: Account):
        # assumes that hist is of shape (hist_window, hist_features)
        # assumes that prediction is of shape (pred_window, pred_features), and that first of pred_feature is close price
        return None

class NaiveLong(Policy):
    def __init__(self):
        super().__init__()
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
    
    def decide(self, symbol:str, hist, price, prediction, account, col_names):
        # print(prediction.shape)
        prediction_window = prediction.shape[0]

        weight_decay = 0.2
        arr = np.ones(prediction_window)
        for i in range(1, prediction_window):
            arr[i] = arr[i-1] * weight_decay
        weights = arr.reshape(prediction_window,1)
        
        weighted_prediction = (prediction * weights).sum() / weights.sum()

        # print(f'prediction: {prediction}, weighted_prediction: {weighted_prediction}')
        
        last_hist = hist[0,-1,:]
        # print(last_hist.shape)
        # print('last_hist: ', last_hist)
        # if prediction[0][0] > 0 and weighted_prediction > 0:
        #     purchase_num = account.place_buy_max_order(symbol, price, 0)
        #     if purchase_num > 0:
        #         account.complete_buy_order(symbol, 0)
        #         # print(f'bought all! {purchase_num} shares at {price}$')
        #         return ('b',purchase_num)
        #     else:
        #         account.cancel_buy_order(symbol, 0)
        #         return ('n',0)
        # elif prediction[0][0] < 0 and weighted_prediction < 0: # (and prediction[0][0] < 0)
        #     purchase_num = account.place_sell_max_order(symbol, price, 0)
        #     if purchase_num > 0:
        #         account.complete_sell_order(symbol, 0)
        #         # print(f'bought all! {purchase_num} shares at {price}$')
        #         return ('s',purchase_num)
        #     else:
        #         account.cancel_sell_order(symbol,0)
        #         return ('n',0)
        # else:
        #     return ('n',0)
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
            purchase_num = account.place_buy_max_order(symbol, price, 0, optimal=False)
            if purchase_num > 0:
                account.complete_buy_order(symbol, 0)
                # print(f'bought all! {purchase_num} shares at {price}$')
                result = ('b',purchase_num)
                self.bought = True
            else:
                print("BANKRUPT!")
                quit()
            #     account.cancel_buy_order(symbol, 0)
            #     result = ('n',0)
        elif pred == -1: # (and prediction[0][0] < 0)
            purchase_num = account.place_sell_max_order(symbol, price, 0)
            if purchase_num > 0:
                account.complete_sell_order(symbol, 0)
                # print(f'bought all! {purchase_num} shares at {price}$')
                result = ('s',purchase_num)
            else:
                account.cancel_sell_order(symbol,0)
                result = ('n',0)


            if weighted_prediction < -0.05: # start short condition
                if self.short == True:
                    return result
                purchase_num = account.place_short_max_order(symbol, price, 0)
                if purchase_num != 0:
                    account.complete_short_order(symbol, 0)
                    # print(f'sold all! {purchase_num} shares at {price}$')
                    result = ('s', result[1] + purchase_num)
                    self.short = True
                else:
                    account.cancel_short_order(symbol,0)
                    # result not changed
        elif pred == -2: # end all short position/sell all long position at the end of day.
            purchase_num = account.place_reverse_short_order(symbol, price,0)
            if purchase_num > 0:
                account.complete_reverse_short_order(symbol, 0)
                # print(f'reversed short! {purchase_num} shares at {price}$')
                result = ('b', purchase_num)
            elif purchase_num == 0:
                # purchase_num = account.cancel_reverse_short_order(symbol, 0) # no buy order is placed
                # result is default result

                # since no need to reverse short position -- we might have positive hold position; sell all of them.
                purchase_num = account.place_sell_max_order(symbol, price, 0)
                if purchase_num > 0:
                    account.complete_sell_order(symbol, 0)
                    # print(f'bought all! {purchase_num} shares at {price}$')
                    result = ('s',purchase_num)
                else:
                    account.cancel_sell_order(symbol,0)
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

        return result

    def get_trade_stat(self):
        # print("long profit pct list: ", self.long_profit_pct_list)
        # print("short profit pct list: ", self.short_profit_pct_list)
        mean_long_profit_pct = statistics.mean(self.long_profit_pct_list)
        mean_short_profit_pct = statistics.mean(self.short_profit_pct_list)
        return self.long_count, self.profitable_long_count, self.short_count, self.profitable_short_count, mean_long_profit_pct, mean_short_profit_pct


def locate_cols(strings_list, substring):
    return [i for i, string in enumerate(strings_list) if substring in string]

class NaiveMACD(Policy):
    def __init__(self, hist_threshold = 0.01):
        super().__init__()
        self.hist_threshold = hist_threshold
        self.bought = False
        self.short = False

        
        self.long_count = 1
        self.profitable_long_count = 0
        self.short_count = 1
        self.profitable_short_count = 0
        self.prev_action_price = 0

        self.last_macd_h = 0
        self.last_macd_list
    
    def decide(self, symbol:str, hist, price, prediction, account, col_names):
        # print("data: ", hist.shape)
        macd_cols = locate_cols(col_names, 'MACD')
        # print("macd_cols: ", macd_cols)
        last_hist = hist[-1,:]
        # print("last_hist: ", last_hist.shape)

        last_macd = last_hist[macd_cols[0]]
        # print("last_macd: ", last_macd)
        last_macd_signal = last_hist[macd_cols[1]]
        last_macd_hist = last_hist[macd_cols[2]]

        edt_scale_col = locate_cols(col_names, 'edt_scaled')[0]
        # print(edt_scale_col)

        edt_scale = last_hist[edt_scale_col]
        # print(edt_scale)

        result = ('n',0) # by default don't do anything


        if last_macd > last_macd_signal:
            pred = 1
        else:
            pred = -1

        if last_macd_hist < self.hist_threshold and last_macd_hist > -self.hist_threshold:
            pred = 0

        if edt_scale < -100:
            return result
        elif edt_scale > 1.6:
            pred = -2 # end of day signal

        
        if pred == 1:
            if self.bought == True:
                # print("Already bought!")
                return result # only invest 0.375 of all balance once.
            purchase_num = account.place_buy_max_order(symbol, price, 0)
            if purchase_num > 0:
                
                self.long_count += 1
                account.complete_buy_order(symbol, 0)
                # print(f'bought all! {purchase_num} shares at {price}$')
                result = ('b',purchase_num)
                self.bought = True
            else:
                print("BANKRUPT!")
                quit()
            #     account.cancel_buy_order(symbol, 0)
            #     result = ('n',0)
        elif pred == -1: # (and prediction[0][0] < 0)
            purchase_num = account.place_sell_max_order(symbol, price, 0)
            if purchase_num > 0:
                account.complete_sell_order(symbol, 0)
                # print(f'bought all! {purchase_num} shares at {price}$')
                result = ('s',purchase_num)
            else:
                account.cancel_sell_order(symbol,0)
                result = ('n',0)


            # if last_macd_hist < -0.1: # start short condition
            #     if self.short == True:
            #         return result
            #     purchase_num = account.place_short_max_order(symbol, price, 0)
            #     if purchase_num != 0:
            #         account.complete_short_order(symbol, 0)
            #         # print(f'sold all! {purchase_num} shares at {price}$')
            #         result = ('s', result[1] + purchase_num)
            #         self.short = True
            #     else:
            #         account.cancel_short_order(symbol,0)
            #         # result not changed
        elif pred == -2: # end all short position/sell all long position at the end of day.
            purchase_num = account.place_reverse_short_order(symbol, price,0)
            if purchase_num > 0:
                account.complete_reverse_short_order(symbol, 0)
                # print(f'reversed short! {purchase_num} shares at {price}$')
                result = ('b', purchase_num)
            elif purchase_num == 0:
                # purchase_num = account.cancel_reverse_short_order(symbol, 0) # no buy order is placed
                # result is default result

                # since no need to reverse short position -- we might have positive hold position; sell all of them.
                purchase_num = account.place_sell_max_order(symbol, price, 0)
                if purchase_num > 0:
                    account.complete_sell_order(symbol, 0)
                    # print(f'bought all! {purchase_num} shares at {price}$')
                    result = ('s',purchase_num)
                else:
                    account.cancel_sell_order(symbol,0)
                    # result = ('n',0)
                    # result is as default
            elif purchase_num < 0:
                print("bankrupt!")
                quit()

        # if result[0] == 'b':
            
        #     if self.short == True:
        #         self.short_count += 1
        #         if self.prev_action_price > price:
        #             self.profitable_short_count += 1
        #         self.short = False

        #     self.prev_action_price = price
        if result[0] == 's':

            if self.bought == True:
                if self.prev_action_price < price:
                    self.profitable_long_count += 1
                self.bought = False

            self.prev_action_price = price

        return result

    def get_trade_stat(self):
        return self.long_count, self.profitable_long_count, self.short_count, self.profitable_short_count


class PolicyComplex(Policy):
    def __init__(self, hist_window, hist_features, pred_window, pred_features):
        super().__init__(hist_window, hist_features, pred_window, pred_features)
    
    def decide(self, symbol:str, hist, price, prediction, account):
        pass

    # TODO: cancel order if waiting too long (after 20s?)


if __name__ == '__main__':
    long = NaiveLong()
    acc = Account(100000, ['BABA'])
    
    price = 100
    prediction = np.arange(101,111).reshape(10,1)
    
    decision = long.decide('BABA', None, price, prediction, acc)
    print(decision)
    print(acc.holding)