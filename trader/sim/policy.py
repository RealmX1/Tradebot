from .account import Account
import numpy as np
import torch

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
    
    def decide(self, symbol:str, hist, price, prediction, account):
        mean_prediction = prediction.mean()
        if prediction[0][0] > 0 and mean_prediction > 0:
            purchase_num = account.place_buy_max_order(symbol, price, 0)
            if purchase_num > 0:
                account.complete_buy_order(symbol, 0)
                # print(f'bought all! {purchase_num} shares at {price}$')
                return ('b',purchase_num)
            else:
                account.cancel_buy_order(symbol, 0)
                return ('n',0)
        elif prediction[0][0] < 0 and mean_prediction < 0: # (and prediction[0][0] < 0)
            purchase_num = account.place_sell_max_order(symbol, price, 0)
            if purchase_num > 0:
                account.complete_sell_order(symbol, 0)
                # print(f'bought all! {purchase_num} shares at {price}$')
                return ('s',purchase_num)
            else:
                account.cancel_sell_order(symbol,0)
                return ('n',0)
        else:
            return ('n',0)
        



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